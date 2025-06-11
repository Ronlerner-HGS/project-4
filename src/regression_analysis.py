import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go
import warnings
import config
warnings.filterwarnings('ignore')

class StudyHoursPredictor:
    def __init__(self, data_path):
        self.my_data = pd.read_csv(data_path)
        self.model_stuff = {}
        self.scores_dict = {}
        self.important_features = {}
        self.subject_encoder = LabelEncoder()
        self.method_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
    def explore_data(self):
        """look at the data"""
        print("=== EXPLORATORY DATA ANALYSIS ===")
        print(f"Dataset shape: {self.my_data.shape}")
        print(f"\nMissing values:\n{self.my_data.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.my_data.describe()}")
        
        # make some graphs
        self._create_eda_plots()
        
    def _create_eda_plots(self):
        """make the plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # hours histogram
        axes[0,0].hist(self.my_data['study_hours_needed'], bins=30, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Distribution of Study Hours Needed')
        axes[0,0].set_xlabel('Hours')
        
        # hours by subject
        sns.boxplot(data=self.my_data, x='subject', y='study_hours_needed', ax=axes[0,1])
        axes[0,1].set_title('Study Hours by Subject')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # correlation thing
        number_columns = self.my_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.my_data[number_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,2])
        axes[0,2].set_title('Feature Correlation Matrix')
        
        # grade scatter plot
        axes[1,0].scatter(self.my_data['previous_grade'], self.my_data['study_hours_needed'], alpha=0.6)
        axes[1,0].set_xlabel('Previous Grade')
        axes[1,0].set_ylabel('Study Hours Needed')
        axes[1,0].set_title('Grade vs Study Hours')
        
        # difficulty scatter
        axes[1,1].scatter(self.my_data['difficulty_rating'], self.my_data['study_hours_needed'], alpha=0.6, color='orange')
        axes[1,1].set_xlabel('Difficulty Rating')
        axes[1,1].set_ylabel('Study Hours Needed')
        axes[1,1].set_title('Difficulty vs Study Hours')
        
        # efficiency plot
        axes[1,2].scatter(self.my_data['study_efficiency'], self.my_data['study_hours_needed'], alpha=0.6, color='green')
        axes[1,2].set_xlabel('Study Efficiency')
        axes[1,2].set_ylabel('Study Hours Needed')
        axes[1,2].set_title('Efficiency vs Study Hours')
        
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_PATH}eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def prepare_features(self):
        """fix up the data for ML"""
        data_processed = self.my_data.copy()
        
        # turn text into numbers
        data_processed['subject_encoded'] = self.subject_encoder.fit_transform(data_processed['subject'])
        data_processed['study_method_encoded'] = self.method_encoder.fit_transform(data_processed['study_method'])
        
        # make some new features
        data_processed['preparedness_score'] = (data_processed['content_familiarity'] + 
                                            data_processed['confidence_level']) / 2
        data_processed['stress_efficiency_ratio'] = data_processed['stress_level'] / data_processed['study_efficiency']
        data_processed['grade_difficulty_interaction'] = data_processed['previous_grade'] * data_processed['difficulty_rating']
        
        # pick which columns to use
        my_features = [
            'subject_encoded', 'previous_grade', 'difficulty_rating', 
            'study_efficiency', 'time_management', 'content_familiarity',
            'stress_level', 'sleep_hours', 'days_available', 'exam_weight',
            'confidence_level', 'study_method_encoded', 'preparedness_score',
            'stress_efficiency_ratio', 'grade_difficulty_interaction'
        ]
        
        X = data_processed[my_features].values
        y = data_processed['study_hours_needed'].values
        
        return X, y
    
    def train_models(self, X, y):
        """train all the different models"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # make everything the same scale
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        # basic linear model
        self.model_stuff['linear'] = LinearRegression()
        self.model_stuff['linear'].fit(X_train_scaled, y_train)
        
        # fancy polynomial one
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train_scaled)
        X_test_poly = poly_features.transform(X_test_scaled)
        
        self.model_stuff['polynomial'] = LinearRegression()
        self.model_stuff['polynomial'].fit(X_train_poly, y_train)
        self.poly_features = poly_features
        
        # ridge thing
        self.model_stuff['ridge'] = Ridge(alpha=1.0)
        self.model_stuff['ridge'].fit(X_train_scaled, y_train)
        
        # random forest (the good one)
        self.model_stuff['random_forest'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model_stuff['random_forest'].fit(X_train, y_train)
        
        # test how good they are
        self._evaluate_models(X_train_scaled, X_test_scaled, X_train_poly, X_test_poly, 
                             X_train, X_test, y_train, y_test)
        
    def _evaluate_models(self, X_train_scaled, X_test_scaled, X_train_poly, X_test_poly,
                        X_train, X_test, y_train, y_test):
        """see how good each model is"""
        
        # linear model results
        y_pred_linear = self.model_stuff['linear'].predict(X_test_scaled)
        self.scores_dict['linear'] = {
            'r2': r2_score(y_test, y_pred_linear),
            'mse': mean_squared_error(y_test, y_pred_linear),
            'mae': mean_absolute_error(y_test, y_pred_linear)
        }
        
        # polynomial results
        y_pred_poly = self.model_stuff['polynomial'].predict(X_test_poly)
        self.scores_dict['polynomial'] = {
            'r2': r2_score(y_test, y_pred_poly),
            'mse': mean_squared_error(y_test, y_pred_poly),
            'mae': mean_absolute_error(y_test, y_pred_poly)
        }
        
        # ridge results
        y_pred_ridge = self.model_stuff['ridge'].predict(X_test_scaled)
        self.scores_dict['ridge'] = {
            'r2': r2_score(y_test, y_pred_ridge),
            'mse': mean_squared_error(y_test, y_pred_ridge),
            'mae': mean_absolute_error(y_test, y_pred_ridge)
        }
        
        # random forest results
        y_pred_rf = self.model_stuff['random_forest'].predict(X_test)
        self.scores_dict['random_forest'] = {
            'r2': r2_score(y_test, y_pred_rf),
            'mse': mean_squared_error(y_test, y_pred_rf),
            'mae': mean_absolute_error(y_test, y_pred_rf)
        }
        
        # what features matter most
        feature_names = [
            'subject', 'previous_grade', 'difficulty_rating', 
            'study_efficiency', 'time_management', 'content_familiarity',
            'stress_level', 'sleep_hours', 'days_available', 'exam_weight',
            'confidence_level', 'study_method', 'preparedness_score',
            'stress_efficiency_ratio', 'grade_difficulty_interaction'
        ]
        
        self.important_features['random_forest'] = dict(zip(
            feature_names, self.model_stuff['random_forest'].feature_importances_
        ))
        
        self._plot_model_comparison()
        self._plot_predictions(X_test_scaled, X_test_poly, X_test, y_test)
        
    def _plot_model_comparison(self):
        """make charts to compare models"""
        models_list = list(self.scores_dict.keys())
        r2_values = [self.scores_dict[model]['r2'] for model in models_list]
        mse_values = [self.scores_dict[model]['mse'] for model in models_list]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R2 chart
        bars1 = ax1.bar(models_list, r2_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax1.set_title('Model Comparison - R² Score')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1)
        
        for bar, score in zip(bars1, r2_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # error chart
        bars2 = ax2.bar(models_list, mse_values, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        ax2.set_title('Model Comparison - Mean Squared Error')
        ax2.set_ylabel('MSE')
        
        for bar, score in zip(bars2, mse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{score:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_PATH}model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_predictions(self, X_test_scaled, X_test_poly, X_test, y_test):
        """show actual vs predicted scatter plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # linear model plot
        y_pred_linear = self.model_stuff['linear'].predict(X_test_scaled)
        axes[0,0].scatter(y_test, y_pred_linear, alpha=0.6)
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Hours')
        axes[0,0].set_ylabel('Predicted Hours')
        axes[0,0].set_title(f'Linear Regression (R² = {self.scores_dict["linear"]["r2"]:.3f})')
        
        # polynomial plot
        y_pred_poly = self.model_stuff['polynomial'].predict(X_test_poly)
        axes[0,1].scatter(y_test, y_pred_poly, alpha=0.6, color='orange')
        axes[0,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0,1].set_xlabel('Actual Hours')
        axes[0,1].set_ylabel('Predicted Hours')
        axes[0,1].set_title(f'Polynomial Regression (R² = {self.scores_dict["polynomial"]["r2"]:.3f})')
        
        # ridge plot
        y_pred_ridge = self.model_stuff['ridge'].predict(X_test_scaled)
        axes[1,0].scatter(y_test, y_pred_ridge, alpha=0.6, color='green')
        axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1,0].set_xlabel('Actual Hours')
        axes[1,0].set_ylabel('Predicted Hours')
        axes[1,0].set_title(f'Ridge Regression (R² = {self.scores_dict["ridge"]["r2"]:.3f})')
        
        # random forest plot
        y_pred_rf = self.model_stuff['random_forest'].predict(X_test)
        axes[1,1].scatter(y_test, y_pred_rf, alpha=0.6, color='purple')
        axes[1,1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1,1].set_xlabel('Actual Hours')
        axes[1,1].set_ylabel('Predicted Hours')
        axes[1,1].set_title(f'Random Forest (R² = {self.scores_dict["random_forest"]["r2"]:.3f})')
        
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_PATH}prediction_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_feature_importance(self):
        """show which features are most important"""
        importance_data = pd.DataFrame(
            list(self.important_features['random_forest'].items()),
            columns=['Feature', 'Importance']
        ).sort_values('Importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(importance_data['Feature'], importance_data['Importance'])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance - Random Forest Model')
        plt.tight_layout()
        plt.savefig(f'{config.FIGURES_PATH}feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def predict_user_study_hours(self, user_data):
        """predict how many hours someone needs to study"""
        # turn user answers into numbers
        user_features = self._prepare_user_features(user_data)
        
        predictions_dict = {}
        
        # try each model
        user_features_scaled = self.feature_scaler.transform([user_features])
        predictions_dict['linear'] = self.model_stuff['linear'].predict(user_features_scaled)[0]
        
        # polynomial one
        user_features_poly = self.poly_features.transform(user_features_scaled)
        predictions_dict['polynomial'] = self.model_stuff['polynomial'].predict(user_features_poly)[0]
        
        # ridge one
        predictions_dict['ridge'] = self.model_stuff['ridge'].predict(user_features_scaled)[0]
        
        # random forest one
        predictions_dict['random_forest'] = self.model_stuff['random_forest'].predict([user_features])[0]
        
        # combine them all (weighted by how good they are)
        weight_values = {model: self.scores_dict[model]['r2'] for model in predictions_dict.keys()}
        total_weight = sum(weight_values.values())
        ensemble_prediction = sum(pred * weight_values[model] / total_weight 
                                for model, pred in predictions_dict.items())
        
        predictions_dict['ensemble'] = ensemble_prediction
        
        return predictions_dict
        
    def _prepare_user_features(self, user_data):
        """turn user answers into the right format"""
        # find subject number
        subject_classes = self.subject_encoder.classes_
        subject_encoded = None
        for i, cls in enumerate(subject_classes):
            if cls == user_data['subject']:
                subject_encoded = i
                break
        
        # find method number
        method_classes = self.method_encoder.classes_
        method_encoded = None
        for i, cls in enumerate(method_classes):
            if cls == user_data['study_method']:
                method_encoded = i
                break
        
        # make combo features
        prep_score = (user_data['content_familiarity'] + user_data['confidence_level']) / 2
        stress_efficiency_thing = user_data['stress_level'] / user_data['study_efficiency']
        grade_difficulty_combo = user_data['previous_grade'] * user_data['difficulty_rating']
        
        return [
            subject_encoded, user_data['previous_grade'], user_data['difficulty_rating'],
            user_data['study_efficiency'], user_data['time_management'], user_data['content_familiarity'],
            user_data['stress_level'], user_data['sleep_hours'], user_data['days_available'],
            user_data['exam_weight'], user_data['confidence_level'], method_encoded,
            prep_score, stress_efficiency_thing, grade_difficulty_combo
        ]
        
    def generate_report(self):
        """make a report about everything"""
        report_lines = []
        report_lines.append("=== IB STUDY HOURS PREDICTION ANALYSIS REPORT ===\n")
        
        report_lines.append("1. DATASET OVERVIEW")
        report_lines.append(f"   - Total records: {len(self.my_data)}")
        report_lines.append(f"   - Features: {len(self.my_data.columns) - 1}")
        report_lines.append(f"   - Target variable: study_hours_needed")
        report_lines.append(f"   - Average study hours: {self.my_data['study_hours_needed'].mean():.1f}")
        report_lines.append(f"   - Study hours range: {self.my_data['study_hours_needed'].min():.1f} - {self.my_data['study_hours_needed'].max():.1f}\n")
        
        report_lines.append("2. MODEL PERFORMANCE COMPARISON")
        for model, scores in self.scores_dict.items():
            report_lines.append(f"   {model.upper()}:")
            report_lines.append(f"     - R² Score: {scores['r2']:.4f}")
            report_lines.append(f"     - MSE: {scores['mse']:.2f}")
            report_lines.append(f"     - MAE: {scores['mae']:.2f}")
        
        report_lines.append("\n3. TOP 5 MOST IMPORTANT FEATURES")
        importance_sorted = sorted(self.important_features['random_forest'].items(), 
                                 key=lambda x: x[1], reverse=True)[:5]
        for i, (feature, importance) in enumerate(importance_sorted, 1):
            report_lines.append(f"   {i}. {feature}: {importance:.4f}")
            
        return "\n".join(report_lines)