import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DATA_PATH = 'data/ib_study_hours_dataset.csv'
OUTPUT_PATH = 'output/'
FIGURES_PATH = 'output/figures/'
RESULTS_PATH = 'output/results/'