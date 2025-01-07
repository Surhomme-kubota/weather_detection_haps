import cv2
import os
import numpy as np
from pathlib import Path

BASEPATH = Path(__file__).resolve().parent.parent


def make_sharp_kernel(k: int) -> np.ndarray:
    return np.array([
        [-k / 9, -k / 9, -k / 9],
        [-k / 9, 1 + 8 * k / 9, -k / 9],
        [-k / 9, -k / 9, -k / 9]
    ], np.float32)


def process_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (224, 224))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    kernel = make_sharp_kernel(1)
    return cv2.filter2D(image, -1, kernel).astype("uint8")


def save_image(image: np.ndarray, output_path: Path, filename: str):
    """
    画像を指定されたパスに保存する。

    Parameters:
    image (np.ndarray): 保存する画像。
    output_path (Path): 保存先のディレクトリパス。
    filename (str): 保存するファイル名。
    """
    if not output_path.exists():
        os.makedirs(output_path)
    cv2.imwrite(str(output_path / filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


class ImageProcessing:
    def __init__(self, filenames: list):
        """
        ImageProcessing class constructor.

        Args:
            filenames (list): List of image file paths to process.
        """
        self.filenames = filenames

    def process(self, data: any = None) -> None:
        """
        Process a list of image files and save them to the specified directory.

        Args:
            data (any): Placeholder for the initial data passed through the pipeline.

        Returns:
            None
        """
        for filename in self.filenames:
            folder = Path(filename).resolve().parent.parent.name
            sub_folder = Path(filename).parent.name
            name = Path(filename).name

            image = cv2.imread(filename)
            processed_image = process_image(image)

            output_path = BASEPATH / 'data' / 'treated_image'
            save_image(processed_image, output_path, name)