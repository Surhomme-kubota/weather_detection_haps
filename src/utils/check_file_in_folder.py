import os


def check_files_in_folders(folders: list):
    """
    指定されたフォルダリストの各フォルダに1つのファイルのみが存在するかを確認する。
    
    :param folders: チェックするフォルダのリスト
    :raises Exception: ファイル数が1でない場合にエラーを発生させる
    """
    for folder in folders:
        # フォルダ内のファイルリストを取得
        try:
            files = os.listdir(folder)
        except FileNotFoundError:
            raise FileNotFoundError(f"{folder} フォルダが存在しません。")

        # ファイルが1つでなければエラーを発生
        if len(files) != 1:
            raise Exception(f"{folder} フォルダにはファイルが1つではありません。現在のファイル数: {len(files)}")

    print("各フォルダには正確に1つのファイルが存在します。")
