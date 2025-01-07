import os
from pathlib import Path

# Define the base path as the grandparent directory of this script file
BASEPATH = Path(__file__).resolve().parent.parent

def make_directory():
    """Create necessary directories for the project."""
    # Define directory paths
    directories = [
        BASEPATH / 'data' / 'raw',
        BASEPATH / 'data' / 'mask',
        BASEPATH / 'data' / 'treated_image',
        BASEPATH / 'data' / 'raw' / 'long',
        BASEPATH / 'data' / 'raw' / 'short',
        BASEPATH / 'results' / 'masked_picture',
        BASEPATH / 'results' / 'masked_thick_cloud',
        BASEPATH / 'results' / 'pred_result',
    ]

    # Create each directory if it doesn't already exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    