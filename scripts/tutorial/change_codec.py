import subprocess
import argparse
from argus import load_model
from argus.callbacks import (
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR,
    LambdaLR,
)
import subprocess
import re
import sys
from src.utils import load_weights_from_pretrain, get_best_model_path, get_lr, get_video_info
from src.ball_action import constants
from src.action.constants import experiments_dir as action_experiments_dir


RESOLUTION = 'square_video'
RESOLUTION_OUT = 'square_video_codec'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--folds", default="all", type=str)
    return parser.parse_args()

def convert_to_h264(input_file, output_file):
    # ffmpegコマンドを使用して、mpeg4コーデックをh264に変換
    cmd = [
        "ffmpeg", "-i", input_file, "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", "-progress", "-", output_file
    ]
    # コマンドを実行
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # 進捗を表示
    duration = None
    time_pattern = re.compile(r"time=(\d+:\d+:\d+\.\d+)")
    duration_pattern = re.compile(r"Duration: (\d+:\d+:\d+\.\d+)")
    for line in process.stdout:
        sys.stdout.write(line)  # 各行を標準出力に書き出す
        # 動画全体の時間を取得
        if duration is None:
            duration_match = duration_pattern.search(line)
            if duration_match:
                duration = duration_match.group(1)
                print(f"Total duration: {duration}")
        # 現在の進捗時間を取得
        time_match = time_pattern.search(line)
        if time_match:
            current_time = time_match.group(1)
            print(f"Current time: {current_time}")
    # プロセスの終了を待つ
    process.wait()
    # 結果を確認
    if process.returncode == 0:
        print(f"Successfully converted {input_file} to {output_file} with h264 codec.")
    else:
        print(f"Failed to convert {input_file}. Error: {process.stderr}")


def detect_fold(experiment: str, fold):
    experiment_dir = constants.experiments_dir / experiment / f"fold_{fold}"
    model_path = get_best_model_path(experiment_dir)
    print(f"Detect games: {experiment=}, {fold=}")
    print("Model path:", model_path)
    data_split = "cv"
    games = constants.fold2games[fold]
    prediction_dir = constants.predictions_dir / experiment / data_split / f"fold_{fold}"
    if not prediction_dir.exists():
        prediction_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {prediction_dir} already exists.")
    for game in games:
        game_dir = constants.soccernet_dir / game
        game_prediction_dir = prediction_dir / game
        game_prediction_dir.mkdir(parents=True, exist_ok=True)
        print("Detect game:", game)
        video_path = game_dir / f"{RESOLUTION}.{constants.videos_extension}" #f"{half}_{RESOLUTION}.mkv"
        video_info = get_video_info(video_path)
        print("Video info:", video_info)
        print("Detect video:", video_path)
        input_video = video_path  # 元のmpeg4ビデオファイル
        output_video = game_dir / f"{RESOLUTION_OUT}.{constants.videos_extension}"  # 変換後のh264ビデオファイル
        convert_to_h264(input_video, output_video)

if __name__ == "__main__":

    args = parse_arguments()

    experiments_dir = constants.experiments_dir / args.experiment

    if args.folds == "all":
        folds = constants.folds
    else:
        folds = [int(fold) for fold in args.folds.split(",")]

    for fold in folds:
        fold_experiment_dir = experiments_dir / f"fold_{fold}"
        print(f"Fold experiment dir: {fold_experiment_dir}")
        detect_fold(args.experiment, fold)