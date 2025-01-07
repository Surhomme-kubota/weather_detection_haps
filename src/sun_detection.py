import torch
from ultralytics import YOLO
import cv2
from typing import List, Tuple

def sun_detect_objects(model_path: str, image_path: str) -> Tuple[float, float, float, float, float, int]:
    """
    YOLOモデルを使用して画像内のオブジェクトを検出し、最も面積が大きなオブジェクトのバウンディングボックス情報を返す関数。
    
    Parameters:
    - model_path (str): トレーニング済みモデルのファイルパス
    - image_path (str): 推論を行いたい画像のパス
    
    Returns:
    Tuple[float, float, float, float, float, int]: 最も面積が大きなオブジェクトに対するバウンディングボックス座標(x1, y1, x2, y2)、信頼度、クラスIDのタプル。
    """
    
    # GPUが使用可能か確認
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # トレーニング済みモデルのロード
    model = YOLO(model_path).to(device)
    
    # 画像を読み込み
    image = cv2.imread(image_path)
    
    # 画像で推論を実行
    results = model.predict(source=image, device=device)
    
    # 推論結果のデータを抽出
    max_area = 0
    best_detection = None
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0].item()
            class_id = box.cls[0].item()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_detection = (x1.item(), y1.item(), x2.item(), y2.item(), confidence, class_id, area)
    
    return best_detection