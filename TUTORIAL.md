# 🎓 IB Study Hours Prediction - Setup Tutorial

Welcome! This tutorial will guide you through setting up and running the IB Study Hours Prediction system on your machine.

This was made with ai


## 📋 Prerequisites

Before you begin, make sure you have:

- **Python 3.8 or higher** installed on your system
- **Git** (to clone the repository)
- **Terminal/Command Prompt** access
- **Internet connection** (for downloading dependencies)

### Check Your Python Version
```bash
python --version
# or
python3 --version
```

If you don't have Python installed, download it from [python.org](https://python.org/downloads/).

---

## 🚀 Quick Start Guide

### Step 1: Clone/Download the Project

If you have the project files, navigate to the project directory:
```bash
cd /path/to/final-project
```

### Step 2: Create a Virtual Environment

This keeps your project dependencies isolated from your system Python:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Linux/macOS)
source venv/bin/activate

# Activate it (Windows)
# venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt when activated.

### Step 3: Install Dependencies

Install all required Python packages:
```bash
pip install -r requirements.txt
```

This will install:
- `pandas` - Data manipulation
- `numpy` - Numerical computations  
- `scikit-learn` - Machine learning
- `matplotlib` & `seaborn` - Data visualization
- `plotly` - Interactive charts
- `openai` - AI assessment (optional)
- `python-dotenv` - Environment variables

### Step 4: Set Up Environment Variables (Optional)

The OpenAI features are optional. If you want to use them:

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```bash
# Open with your preferred editor
nano .env
# or
code .env
```

3. Replace `your_openai_api_key_here` with your actual API key:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Note:** The system works perfectly without OpenAI - it will just skip the AI-powered assessment features.

### Step 5: Run the Program

```bash
python main.py
```

That's it! The program should start and guide you through the assessment.

---

## 📁 Project Structure

```
final-project/
├── main.py                    # Main program - run this!
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── data/
│   └── ib_study_hours_dataset.csv  # Training data
├── src/
│   ├── __init__.py
│   ├── ai_assessment.py       # User questionnaire
│   └── regression_analysis.py # ML models
└── output/                    # Generated results
    ├── figures/              # Charts and plots
    └── results/              # JSON results
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### "Python not found" or "Command not found"
- **Solution**: Make sure Python is installed and added to your PATH
- Try `python3` instead of `python`

#### "pip not found"
- **Solution**: 
```bash
# Try this instead
python -m pip install -r requirements.txt
```

#### Permission errors on Linux/macOS
- **Solution**: Use `python3` and ensure you have write permissions:
```bash
chmod +x main.py
python3 main.py
```

#### Virtual environment activation issues
- **Linux/macOS**:
```bash
source venv/bin/activate
```
- **Windows (Command Prompt)**:
```bash
venv\Scripts\activate.bat
```
- **Windows (PowerShell)**:
```bash
venv\Scripts\Activate.ps1
```

#### Missing dependencies after installation
```bash
# Upgrade pip first
pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### OpenAI API errors
- The system will work without OpenAI
- Check your API key in the `.env` file
- Ensure you have credits in your OpenAI account

---

## 🎯 How to Use the System

### 1. Start the Program
```bash
python main.py
```

### 2. Follow the Interactive Assessment
The system will ask you questions about:
- Which IB subject you're studying
- Your previous grades
- Study habits and preferences
- Confidence levels
- Available study time

### 3. View Your Results
The system will:
- Predict your optimal study hours using 4 different ML models
- Show you visualizations of the analysis
- Save your results to `output/results/user_assessment.json`
- Generate comparison charts in `output/figures/`

### 4. Understand the Output
- **Linear Regression**: Basic linear relationship
- **Polynomial Regression**: Captures non-linear patterns
- **Ridge Regression**: Reduces overfitting
- **Random Forest**: Ensemble method for robust predictions
- **Ensemble**: Combines all models for the best prediction

---

## 📊 What the System Generates

### Files Created:
- `output/figures/eda_analysis.png` - Data exploration visualizations
- `output/figures/model_comparison.png` - Model performance comparison
- `output/figures/prediction_plots.png` - Actual vs predicted scatter plots
- `output/results/user_assessment.json` - Your assessment responses

### Console Output:
- Step-by-step analysis progress
- Model training results
- Your personalized study hour recommendations
- Performance metrics for each model

---

## 🔄 Running Multiple Times

You can run the assessment multiple times:
- Each run will overwrite previous results
- Try different answers to see how predictions change
- The system learns from the dataset, not your individual responses

---

## 🛠️ Advanced Usage

### Run Individual Components

To just train models without assessment:
```python
from src.regression_analysis import StudyHoursPredictor
predictor = StudyHoursPredictor('data/ib_study_hours_dataset.csv')
X, y = predictor.prepare_features()
predictor.train_models(X, y)
```

### Modify the Dataset
- Edit `data/ib_study_hours_dataset.csv` to add your own data
- Follow the same column structure
- Restart the program to use new data

### Customize Output Paths
Edit `config.py` to change where files are saved:
```python
OUTPUT_PATH = 'my_custom_output/'
FIGURES_PATH = 'my_custom_output/charts/'
RESULTS_PATH = 'my_custom_output/data/'
```

---

## 🆘 Getting Help

If you encounter issues:

1. **Check the error message** - it often tells you exactly what's wrong
2. **Verify your Python version** - needs 3.8+
3. **Make sure virtual environment is activated** - look for `(venv)` in prompt
4. **Try reinstalling dependencies**:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```
5. **Check file permissions** - ensure you can read/write in the project directory

---

## 🎉 Success!

If you see output like this, everything is working:
```
Starting IB Study Hours Prediction System...
=== IB STUDY ASSESSMENT ===
Let's gather some information about your study needs.

Question 1/8: Which IB subject are you studying?
...
```

**Congratulations! You're ready to discover your optimal study hours! 📚✨**

---

*made by ai*
