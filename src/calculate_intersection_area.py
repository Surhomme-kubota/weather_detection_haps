import cv2
import numpy as np 



def calculate_intersection_area(mask_path1: str,
                                mask_path2: str) -> int:
    """
    二つのマスク画像の重なる白い部分の面積をピクセル数で返す関数。
    
    Parameters:
    - mask_path1 (str): 一枚目のマスク画像のファイルパス。
    - mask_path2 (str): 二枚目のマスク画像のファイルパス。
    
    Returns:
    int: 重なる部分の面積（ピクセル数）。
    """
    
    # 画像をグレースケールで読み込み
    mask1 = cv2.imread(mask_path1, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)
    
    # 2値化処理 (しきい値を128に設定し、それより大きい値を255 (白), それ以下を0 (黒) にする)
    _, binary_mask1 = cv2.threshold(mask1, 128, 255, cv2.THRESH_BINARY)
    _, binary_mask2 = cv2.threshold(mask2, 128, 255, cv2.THRESH_BINARY)
    
    # 重なる部分の計算（論理積）
    intersection = np.logical_and(binary_mask1 == 255, binary_mask2 == 255)
    
    # 重なる部分の面積（白いピクセルの数）を計算
    area = np.sum(intersection)
    
    return area