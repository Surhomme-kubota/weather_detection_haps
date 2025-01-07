import subprocess
from pathlib import Path

# detection.py スクリプトのパス
script_path = str(Path(__file__).parent / "src" / "detect.py")

# スクリプトを実行するためのコマンド
command = f'python3 {script_path}'

# subprocessを使ってスクリプトを実行
process = subprocess.run(command, shell=True, check=True)

# 実行結果の出力
print("Script exited with status code:", process.returncode)