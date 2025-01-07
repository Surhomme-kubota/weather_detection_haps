import os


class CheckDirectoryContents:
    def __init__(self, directories):
        self.directories = directories

    def process(self, image=None):
        """ディレクトリ内の画像ファイル数を確認する"""
        results = {}
        for directory in self.directories:
            files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
            is_correct = len(files) == 1
            results[directory] = (is_correct, files)
        
        for directory, (is_correct, files) in results.items():
            if is_correct:
                print(f"{directory} ディレクトリは条件を満たしています。ファイル: {files}")
            else:
                print(f"{directory} ディレクトリは条件を満たしていません。ファイル数: {len(files)}, ファイル: {files}")
                raise ValueError("ディレクトリの画像ファイル数に間違いがございます")
        return image