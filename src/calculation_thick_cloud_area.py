import cv2
import numpy as np



def calculate_white_area(mask_path: str) -> int:
    """
    指定されたマスク画像内の白い面積（ピクセル数）を計算します。

    Parameters:
        mask_path (str): マスク画像のファイルパス。

    Returns:
        int: マスク内の白いピクセルの数。

    Raises:
        FileNotFoundError: 指定されたパスのファイルが見つからない場合に発生。

    Example:
        white_area = calculate_white_area('path/to/mask.png')
        print(f"White area in pixels: {white_area}")
    """
    # マスク画像をグレースケールで読み込む
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"The mask file at {mask_path} could not be found.")

    # マスク画像の白い部分（255）のみを抽出するための閾値処理
    _, thresholded_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 白いピクセルの数を数える
    white_area = np.sum(thresholded_mask == 255)

    return white_area