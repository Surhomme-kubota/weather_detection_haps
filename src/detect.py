from utils.flow import Flow
from utils.check_directory_contents import CheckDirectoryContents
from utils.get_filename import get_filenames
from utils.create_datetime_filename import create_datetime_filename
from apply_mask_to_image import ApplyMaskToImage
from image_processing import ImageProcessing
from rain_detection import rain_main
from mask import CreateFisheyeMaskWithAnnotations


from pathlib import Path
import numpy as np 
from glob import glob
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime, date, time


from sun_detection import sun_detect_objects
from draw_box_if_in_mask import is_center_in_white_area
from cloud_detection import cloud_detection_main
from calculate_intersection_area import calculate_intersection_area
from calculation_thick_cloud_area import calculate_white_area
from delete_file import delete_files_in_directories
from white_cloud_detection import white_cloud_detection_main


def get_current_date_and_time() -> tuple[date, time]:
    # 現在の日時を取得
    now = datetime.now()
    
    # 日付と時間に分ける
    current_date = now.date()
    current_time = now.time()
    
    return current_date, current_time


class RainDetection:
    def process(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        雨の検出を行い、結果に基づいてデータを更新する

        Args:
            data (dict, optional): 更新するデータの辞書。デフォルトはNone。

        Returns:
            dict: 更新されたデータの辞書。

        Raises:
            ValueError: rain_detectionの値が不正な場合に発生。
        """
        if data is None:
            data = {}

        # ダミーの雨検出のロジック
        rain_detection = rain_main()
        print('天候:', rain_detection)

        if rain_detection[0] == '晴れまたは曇り':
            rain_num = 0 
            
        elif rain_detection[0] == '雨':
            rain_num = 1
            
        else:
            raise ValueError("rain_detectionに何かしらの値が返されていません")

        # Update data dictionary with new information
        data['rain_num'] = rain_num
        data['rain_detection'] = rain_detection

        return data
    
    
class SunDetection:
    def __init__(self, model_path: str, image_file_directory: str) -> None:
        
        self.model_path = model_path
        self.image_file_directory = image_file_directory
        
    def process(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        太��の検出を行い、結果に基づいてデータを更新する

        Args:
            data (dict, optional): 更新するデータの��書。デフォルトはNone。

        Returns:
            dict: 更新されたデータの��書。

        Raises:
            ValueError: sun_detectionの値が不正な場合に発生。
        """
        
        if data is None:
            data = {}

        # 太陽検出のロジック
        detection = sun_detect_objects(self.model_path, self.image_file_directory)
        print("detection:", detection)
        
        if detection is None:
            print("太陽が検出されませんでした")
            
            data['first_detection_coords'] = []
            data['sun_position_x'] = ''
            data['sun_position_y'] = ''
            data['sun_detection'] = 0
            data['sun_area'] = 0
        
        else:   
            # 太陽位置の検出
            detection_list = [tuple(detection[:4])]
        
            # 最初の検出結果の座標だけを取り出す
            if detection_list:  # 検出リストが空でない場合にのみ処理
                first_detection_coords = detection_list[0]
                
            # すべての検出から面積のみ取り出す
            sun_area = detection[-1].item()
            
            if len(detection_list) == 1:
                print("太陽検出位置:", first_detection_coords)
                print("太陽面積:", sun_area)
                sun_detection = 1
                
            else:
                sun_detection = 0
                sun_area = 0
                print("太陽が検出されませんでした")
                
            x1, y1, x2, y2 = first_detection_coords
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            data['first_detection_coords'] = first_detection_coords
            data['sun_position_x'] = center_x
            data['sun_position_y'] = center_y
            data['sun_detection'] = sun_detection
            data['sun_area'] = sun_area
        
        return data


def image_pipeline():
    
    # Build the pipeline
    flow1 = Flow()
    
    """
    ディレクトリ内の画像枚数チェック
    """
    # Check the amount of pictures in target directories
    long_directory = str(Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'long')
    short_directory = str(Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'short')
    check_directory = [long_directory, short_directory]
    flow1.add_step(CheckDirectoryContents(check_directory))
    
    """
    マスク画像の作成
    """
    long_image_name = get_filenames(long_directory)
    long_image_directory = str(Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'long' / long_image_name[0])
    mask_image_directory = str(Path(__file__).resolve().parent.parent / 'data' / 'mask' / 'mask_image_function.jpg')
    flow1.add_step(CreateFisheyeMaskWithAnnotations(long_image_directory, 124, 236, 41, 81, mask_image_directory))
    
    """
    長時間露光画像のマスク画像作成
    """
    # Making masked pictures
    masked_image_name = create_datetime_filename(base_name="mask")
    output_image_directory = str(Path(__file__).resolve().parent.parent / 'results' / 'masked_picture' / masked_image_name)
    flow1.add_step(ApplyMaskToImage(long_image_directory, mask_image_directory, output_image_directory))
    
    """
    雨の検出
    """
    # 雨の検出(雨が検出した場合は、雨、それ以外は晴れまたは曇りで返す)
    long_image_path = str(Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'long' / '*.jpg')
    path = glob(long_image_path)
    # 画像の加工
    flow1.add_step(ImageProcessing(path))
    # 雨の検出
    flow1.add_step(RainDetection())
    rain_processed_data = flow1.execute()
    
    # 雨検出時値をまとめる
    if rain_processed_data['rain_num'] == 1:
        # 日時取得
        current_date, current_time = get_current_date_and_time()
        data_to_save = {
                'sun_position_x': np.nan,
                'sun_position_y': np.nan,
                'sun_detection': 0,
                'sun_area': 0,
                'area_sun_detection': 0,
                'duplicated_thick_cloud_area':13655,
                'all_thick_cloud_area': 550000,
                "duplicated_cloud_area": 13655,
                "cloud_area": 550000,
                'rain_detection': 1,
                'date': current_date,
                'time': current_time,
                "masked_picture_name": masked_image_name,
            }
        df = pd.DataFrame([data_to_save])
        df.to_csv(str(Path(__file__).resolve().parent.parent / 'results' / 'pred_result' / 'detection_data.csv'),
                  index=False)
        
    else:
        """
        太陽検出
        """
        # Build instance
        flow2 = Flow()
        
        model_path = Path(__file__).resolve().parent.parent / 'models' / 'sun' / 'best.pt'
        short_image_name = get_filenames(short_directory)
        short_image_directory = str(Path(__file__).parent.parent / 'data' / 'raw' / 'short' / short_image_name[0])
        flow2.add_step(SunDetection(model_path, short_image_directory))
        sun_processed_data = flow2.execute()
        
        if sun_processed_data['sun_detection'] == 1:
            sun_position = sun_processed_data['first_detection_coords']
            number = is_center_in_white_area(mask_image_directory, sun_position)

            if number == 1:
                sun_processed_data['area_sun_detection'] = number 
                print("所定範囲内に太陽検出の有無: 検出")
            
            else:
                sun_processed_data['area_sun_detection'] = number
                print("所定範囲内に太陽検出の有無: 未検出")
        else:
            sun_processed_data['area_sun_detection'] = 0
        
            
        """
        厚い雲と通常雲の検出
        """
        # 厚い雲のエリアマスク画像出力
        cloud_detection_main()
        
        # 雲全体のマスク画像出力
        white_cloud_detection_main()
        
        # 作成した厚い雲のマスクディレクトリ
        cloud_mask_folder_directory = str(Path(__file__).resolve().parent.parent / 'results' / 'masked_thick_cloud')
        cloud_mask_image_name = get_filenames(cloud_mask_folder_directory)
        cloud_mask_directory = str(Path(__file__).resolve().parent.parent / 'results' / 'masked_thick_cloud' / cloud_mask_image_name[0])
        
        # 作成した通常雲のマスクディレクトリ
        white_mask_folder_directory = str(Path(__file__).resolve().parent.parent / 'results' / 'masked_cloud')
        white_cloud_mask_image_name = get_filenames(white_mask_folder_directory)
        white_cloud_mask_directory = str(Path(__file__).resolve().parent.parent / 'results' / 'masked_cloud' / white_cloud_mask_image_name[0])
        
        # maskの枚数が適正か否かをチェック
        if len(cloud_mask_image_name) != 1:
            raise ValueError("厚い雲マスクのファイル数が適切ではありません")
        
         # maskの枚数が適正か否かをチェック
        if len(white_cloud_mask_image_name) != 1:
            raise ValueError("通常雲マスクのファイル数が適切ではありません")
        
        # 厚い雲と所定エリアの重なり面積算出
        duplicated_cloud_area = calculate_intersection_area(output_image_directory, cloud_mask_directory)
        sun_processed_data['duplicated_thick_cloud_area'] = duplicated_cloud_area
        print("所定エリア内の厚雲面積:", duplicated_cloud_area)
        
        # 画像全体の厚い雲面積算出
        white_area = calculate_white_area(cloud_mask_directory)
        print("画像内全体の厚雲面積:", white_area)
        sun_processed_data['all_thick_cloud_area'] = white_area
        
        # 通常の雲と所定エリアの量な理面積算出
        white_duplicated_cloud_area = calculate_intersection_area(output_image_directory, white_cloud_mask_directory)
        sun_processed_data['duplicated_cloud_area'] = white_duplicated_cloud_area
        print("所定エリア内の通常雲面積:", white_duplicated_cloud_area)
        
        # 画像全体の通常雲面積算出
        white_cloud_area = calculate_white_area(white_cloud_mask_directory)
        print("画像内全体の雲面積:", white_cloud_area)
        sun_processed_data['cloud_area'] = white_cloud_area
        
        # 雨の検出データ格納
        sun_processed_data['rain_detection'] = 0
        
        # 日時取得
        current_date, current_time = get_current_date_and_time()
        sun_processed_data['date'] = current_date
        sun_processed_data['time'] = current_time
        
        # マスクイメージ画像格納
        sun_processed_data['masked_picture_name'] = masked_image_name
        del sun_processed_data['first_detection_coords']
        
        # Save data file
        df = pd.DataFrame([sun_processed_data])
        df.to_csv(str(Path(__file__).resolve().parent.parent / 'results' / 'pred_result' / 'detection_data.csv'),
                  index=False)
        
    """
    データ収集済みファイル削除
    """
    mask_directory = Path(__file__).resolve().parent.parent / "data" / "mask"
    masked_image_directory = Path(__file__).resolve().parent.parent / "results" / "masked_thick_cloud"
    treated_image_directory = Path(__file__).resolve().parent.parent / "data" / "treated_image"
    white_masked_image_directory = Path(__file__).resolve().parent.parent / "results" / "masked_cloud"
    masked_picture_directory = Path(__file__).resolve().parent.parent / "results" / "masked_picture"
    
    directories_to_clean = [
                            long_directory,
                            short_directory,
                            mask_directory,
                            treated_image_directory,
                            masked_image_directory,
                            white_masked_image_directory,
                            masked_picture_directory,
                            ]
    
    # delete_files_in_directories(directories_to_clean)
        

if __name__ == "__main__":
    
    image_pipeline()