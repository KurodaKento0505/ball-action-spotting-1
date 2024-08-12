import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

import time
import json
import argparse
import multiprocessing
from pathlib import Path
from pprint import pprint
from importlib.machinery import SourceFileLoader

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

import torch
import torch._dynamo
torch.cuda.empty_cache()

from argus import load_model
from argus.callbacks import (
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR,
    LambdaLR,
)

from src.ball_action.annotations import get_videos_data, get_videos_sampling_weights
from src.utils import load_weights_from_pretrain, get_best_model_path, get_lr, get_video_info
from src.predictors import MultiDimStackerPredictor
from src.frame_fetchers import NvDecFrameFetcher
from src.data_loaders import RandomSeekDataLoader, SequentialDataLoader
from src.ball_action.augmentations import get_train_augmentations
from src.indexes import StackIndexesGenerator, FrameIndexShaker
from src.datasets import TrainActionDataset, ValActionDataset
from src.metrics import AveragePrecision, Accuracy
from src.target import MaxWindowTargetsProcessor
from src.argus_models import BallActionModel
from src.ema import ModelEma, EmaCheckpoint
from src.frames import get_frames_processor
from src.ball_action import constants
from src.mixup import TimmMixup
from src.action.constants import experiments_dir as action_experiments_dir


RESOLUTION = "720p"
TTA = True


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--folds", default="all", type=str)
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--challenge", action="store_true")
    parser.add_argument("--use_saved_predictions", action="store_true")
    return parser.parse_args()


def detect_game(predictor: MultiDimStackerPredictor,
                game: str,
                prediction_dir: Path,
                use_saved_predictions: bool):
    game_dir = constants.soccernet_dir / game
    game_prediction_dir = prediction_dir / game
    game_prediction_dir.mkdir(parents=True, exist_ok=True)
    print("Detect game:", game)
    video_path = game_dir / f"{RESOLUTION}.{constants.videos_extension}" #f"{half}_{RESOLUTION}.mkv"
    video_info = get_video_info(video_path)
    print("Video info:", video_info)
    print("Detect video:", video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return
    n = 0
    # YOLOv10モデルの読み込み
    model = YOLO("yolov10n.pt")
    # 動画ライターの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデックの設定
    output = cv2.VideoWriter('data/ball_action/detections/output_video.mp4', fourcc, 25.0, (1280, 720))  # 出力ファイル、フレームレート、フレームサイズを設定
    with tqdm(total = 300) as t:
        while n < 300:
            t.update()
            ret, frame_np = cap.read()
            if ret:
                # save
                # cv2.imwrite('data/ball_action/detections/hsv_frame.png', hsv_frame)

                # グレースケール変換
                gray_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)

                # 閾値処理（白いラインやボールを強調）
                _, binary_frame = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)
                
                # カラーを元にした選手検出
                hsv_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2HSV)

                # ピッチ以外を黒色に
                # 緑色の範囲を定義 (この範囲は調整が必要です)
                lower_green = np.array([30, 40, 40])
                upper_green = np.array([90, 255, 255])
                # 緑色のマスクを作成
                mask = cv2.inRange(hsv_frame, lower_green, upper_green)
                # マスクを使って元の画像からピッチ部分を抽出
                result = cv2.bitwise_and(frame_np, frame_np, mask=mask)
                # 輪郭を検出
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # マスクを初期化（全て黒色）
                mask_with_contour = np.zeros_like(frame_np)
                # 輪郭を白色で描画
                cv2.drawContours(mask_with_contour, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
                # マスクの部分だけ元の画像を抽出（他は黒）
                pitch = cv2.bitwise_and(frame_np, mask_with_contour)
                cv2.imwrite('data/ball_action/detections/pitch.png', pitch)

                # 白色（ラインやボール）のマスク
                white_mask = cv2.inRange(hsv_frame, np.array([0, 0, 200]), np.array([180, 50, 255]))

                # 色の範囲に基づいて選手のマスクを作成（例: 青色と赤色のユニフォーム）
                blue_mask = cv2.inRange(hsv_frame, np.array([100, 150, 0]), np.array([140, 255, 255]))
                red_mask = cv2.inRange(hsv_frame, np.array([0, 50, 50]), np.array([10, 255, 255]))
                red_mask2 = cv2.inRange(hsv_frame, np.array([170, 50, 50]), np.array([180, 255, 255]))
                player_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(red_mask, red_mask2))

                # 二値化されたライン、選手、ボールを1つのマスクに統合
                combined_mask = cv2.bitwise_or(binary_frame, cv2.bitwise_or(white_mask, player_mask))

                # モルフォロジー変換（ノイズ除去と形状強調）
                kernel = np.ones((5, 5), np.uint8)
                combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

                # マスクを使って元のフレームをフィルタリング
                filtered_frame = cv2.bitwise_and(frame_np, frame_np, mask=combined_mask)
                
                # YOLOv10で物体検出
                results = model(filtered_frame)
                # 検出結果を描画
                result_img = results[0].plot()  # 検出結果を画像に描画
                # 動画にフレームを追加
                output.write(result_img)
                n += 1
                break
            else:
                break
    
    # 動画ファイルを閉じる
    # output.release()
    # cv2.destroyAllWindows()

def detect_fold(experiment: str, fold, gpu_id: int,
                challenge: bool, use_saved_predictions: bool):
    experiment_dir = constants.experiments_dir / experiment / f"fold_{fold}"
    model_path = get_best_model_path(experiment_dir)
    print(f"Detect games: {experiment=}, {fold=}, {gpu_id=} {challenge=}")
    print("Model path:", model_path)
    predictor = MultiDimStackerPredictor(model_path, device=f"cuda:{gpu_id}", tta=TTA)
    if challenge:
        data_split = "challenge"
        games = constants.challenge_games
    else:
        if fold == 'train':
            data_split = 'test'
            games = constants.fold2games[5] + constants.fold2games[6]
        else:
            data_split = "cv"
            games = constants.fold2games[fold]
    prediction_dir = constants.predictions_dir / experiment / data_split / f"fold_{fold}"
    if not prediction_dir.exists():
        prediction_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder {prediction_dir} already exists.")
    for game in games:
        detect_game(predictor, game, prediction_dir, use_saved_predictions)


if __name__ == "__main__":

    args = parse_arguments()

    experiments_dir = constants.experiments_dir / args.experiment

    if args.folds == 'train':
        train_folds = [0, 1, 2, 3, 4]
        val_folds = [5, 6]
        train_games = []
        val_games = constants.fold2games[5] + constants.fold2games[6]

        for train_fold in train_folds:
            train_games += constants.fold2games[train_fold]
        fold_experiment_dir = experiments_dir / f"fold_train"
        print(f"Val folds: {val_folds}, train folds: {train_folds}")
        print(f"Val games: {val_games}, train games: {train_games}")
        print(f"Fold experiment dir: {fold_experiment_dir}")
        detect_fold(args.experiment, "train", args.gpu_id,
                     args.challenge, args.use_saved_predictions)

    else:
        if args.folds == "all":
            folds = constants.folds
        else:
            folds = [int(fold) for fold in args.folds.split(",")]

        for fold in folds:
            train_folds = list(set(constants.folds) - {fold})
            val_games = constants.fold2games[fold]
            train_games = []
            for train_fold in train_folds:
                train_games += constants.fold2games[train_fold]
            fold_experiment_dir = experiments_dir / f"fold_{fold}"
            print(f"Val fold: {fold}, train folds: {train_folds}")
            print(f"Val games: {val_games}, train games: {train_games}")
            print(f"Fold experiment dir: {fold_experiment_dir}")
            detect_fold(args.experiment, fold, args.gpu_id,
                         args.challenge, args.use_saved_predictions)