#!/usr/bin/env python3

# Main file to demo the study hours prediction system
import os
from src.ai_assessment import StudyAssessment
from src.regression_analysis import StudyHoursPredictor
import config

def main():
    # Initialize the assessment
    print("Starting IB Study Hours Prediction System...")
    
    # Create output directories if they don't exist
    os.makedirs(config.FIGURES_PATH, exist_ok=True)
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    
    # Get API key from config
    api_key = config.OPENAI_API_KEY
    if not api_key or api_key == "your_openai_api_key_here":
        print("Warning: Please set your OpenAI API key in the .env file")
        print("For now, using demo mode...")
        api_key = None
    
    # Conduct the assessment
    assessment_thing = StudyAssessment(api_key)
    user_answers = assessment_thing.conduct_assessment()
    
    print("\nGreat! Now let me analyze your responses and predict study hours...\n")
    
    # Load the predictor
    predictor_system = StudyHoursPredictor(config.DATA_PATH)
    
    # Prepare and train models
    features_data, target_data = predictor_system.prepare_features()
    predictor_system.train_models(features_data, target_data)
    
    # Get predictions
    study_predictions = predictor_system.predict_user_study_hours(user_answers)
    
    # Show results
    print("=== STUDY HOURS RECOMMENDATIONS ===")
    for model_name, hours in study_predictions.items():
        print(f"{model_name.title()}: {hours:.1f} hours")
    
    print(f"\n RECOMMENDED STUDY TIME: {study_predictions['ensemble']:.1f} hours")
    
    # Save the results
    assessment_thing.save_responses('user_assessment.json')
    print("\nYour assessment has been saved!")

if __name__ == "__main__":
    main()
