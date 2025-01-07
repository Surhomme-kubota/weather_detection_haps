import datetime


def create_datetime_filename(base_name="Box", extension= "jpg") -> str:
    """
    基本名と拡張子を受け取り、現在の日時を含むファイル名を生成する関数。
    
    Parameters:
    - base_name (str): ファイルの基本名
    - extension (str): ファイルの拡張子
    
    Returns:
    str: 日時が付加されたファイル名
    """
    # 現在の日時を取得
    now = datetime.datetime.now()
    # 日時を文字列にフォーマット（例：20230714_123456）
    datetime_str = now.strftime('%Y%m%d_%H%M%S')
    # ファイル名を組み立て
    filename = f"{base_name}_{datetime_str}.{extension}"
    
    return filename