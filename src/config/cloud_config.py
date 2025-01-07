
from pathlib import Path
from datetime import date

# Constants
ENCODER = 'tf_efficientnetv2_xl'
CLASSES = ['tower']
DEVICE = 'cuda'
EPOCHS = 50
LR = 0.00001
SAVEDATE = date.today()

# Base path
BASEPATH = Path(__file__).resolve().parent.parent.parent

# File and folder paths
DATA_FOLDER = BASEPATH / 'data' / 'raw' / 'long'
RESULTS_FOLDER = BASEPATH / 'results' / 'masked_thick_cloud'
MODEL_FOLDER = BASEPATH / 'models' / 'thick_cloud'
MODEL_FILENAME = '2024-07-14_best_valid_model.pth'

# Paths for model saving and data
TEST_FILE_FOLDER_PATH = DATA_FOLDER
OUTPUT_FILE_FOLDER_PATH = RESULTS_FOLDER
MODEL_PATH = MODEL_FOLDER / MODEL_FILENAME