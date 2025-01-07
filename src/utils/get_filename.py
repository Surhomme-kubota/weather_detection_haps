import os

def get_filenames(directory: str) -> str:
    """
    指定されたディレクトリに存在するすべてのファイルとディレクトリの名前をリストとして返す関数。
    
    Parameters:
    - directory (str): ファイル名を取得したいディレクトリのパス
    
    Returns:
    List[str]: ディレクトリ内のファイルおよびサブディレクトリ名のリスト
    """
    # 指定ディレクトリ内の全てのエントリーを取得
    filenames = os.listdir(directory)
    
    return filenames