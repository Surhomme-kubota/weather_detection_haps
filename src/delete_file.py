import os
from typing import List


def delete_files_in_directories(directories: List[str]) -> None:
    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
    else:
        print(f"Directory does not exist: {directory}")