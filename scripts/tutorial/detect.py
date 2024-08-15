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
from filterpy.kalman import KalmanFilter
from deep_sort_realtime.deepsort_tracker import DeepSort

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

from src.utils import load_weights_from_pretrain, get_best_model_path, get_lr, get_video_info
from src.ball_action import constants
from src.action.constants import experiments_dir as action_experiments_dir

from estimate_camera_movement import Wide_Angle_Shot


RESOLUTION = "720p"
TTA = True


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--folds", default="all", type=str)
    return parser.parse_args()


def tracking(frame, results):
    # DeepSORTの初期化
    tracker = DeepSort(max_age=30,
                        n_init=3,
                        max_iou_distance=0.7,
                        nn_budget=100)

    boxes = results[0].boxes
    detections = []
    # ボールを認識
    if boxes:
        # confidenceが最も高いボールを選択
        best_box = max(boxes, key=lambda x: x.conf)
        x_min, y_min, x_max, y_max = best_box.xyxy[0].tolist()  # 座標を取得
        confidence = best_box.conf.item()  # 信頼度を取得
        width = x_max - x_min
        height = y_max - y_min
        # 座標を整数型に変換
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        yolo_center = [(x_min + x_max) // 2, (y_min + y_max) // 2]
        # DeepSORT用の検出情報を作成
        detection = ([x_min, y_min, width, height], confidence, 0)  # 最後の0はクラスID（今回は0としています）
        detections.append(detection)
    # DeepSORTで追跡
    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if track.is_confirmed() and track.time_since_update <= 1:
            bbox = track.to_ltrb()  # left, top, right, bottom
            track_center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]
            break
    else:
        track_center = None
        yolo_center = None

    if not boxes:
        for track in tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.to_ltrb()
                track_id = track.track_id
                track_center = [(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2]

    # YOLOの検出結果とDeepSORTの予測結果を比較
    if previous_ball_position and track_center:
        yolo_distance = ((yolo_center[0] - previous_ball_position[0]) ** 2 +
                        (yolo_center[1] - previous_ball_position[1]) ** 2) ** 0.5
        track_distance = ((track_center[0] - previous_ball_position[0]) ** 2 +
                        (track_center[1] - previous_ball_position[1]) ** 2) ** 0.5

        if yolo_distance < track_distance:
            final_bbox = [x_min, y_min, x_max, y_max]  # YOLOの結果を使用
            previous_ball_position = yolo_center
        else:
            final_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # DeepSORTの結果を使用
            previous_ball_position = track_center

    elif track_center:
        final_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # DeepSORTの結果を使用
        previous_ball_position = track_center

    elif yolo_center:
        final_bbox = [x_min, y_min, x_max, y_max]  # YOLOの結果を使用
        previous_ball_position = yolo_center

    else:
        final_bbox = None  # ボールが検出されなかった
    
    # 正しい検出結果を描画
    if final_bbox:
        cv2.rectangle(frame, (int(final_bbox[0]), int(final_bbox[1])),
                    (int(final_bbox[2]), int(final_bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, "Ball", (int(final_bbox[0]), int(final_bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    '''# YOLOの検出結果を描画
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    cv2.putText(frame, "YOLO", (x_center - 10, y_center - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)'''

        
    # ボールが検出されなかった場合でもDeepSORTの結果を描画
    if not boxes:
        for track in tracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.to_ltrb()
                track_id = track.track_id
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# フレームを切り取る
def cut_frame(frame):
    # フレームのサイズ取得
    h, w, _ = frame.shape
    # フレームの中央部分を切り取る（幅、高さを半分に）
    cut_frame = frame[h//4:3*h//4, w//4:3*w//4]
    return cut_frame


# ボール周辺以外にマスクをかける
def mask(frame, results):
    # フレームを黒くする
    masked_frame = np.zeros_like(frame)
    # bbox
    boxes = results[0].boxes
    # confidenceが最も高いボールを選択
    best_box = max(boxes, key=lambda x: x.conf)
    x_min, y_min, x_max, y_max = best_box.xyxy[0].tolist()  # 座標を取得
    x_center = (x_min + x_max)/2
    y_center = (y_min + y_max)/2
    # ボールの周り360x640の矩形を白くする
    x1 = max(0, x_center - 320)
    y1 = max(0, y_center - 180)
    x2 = min(frame.shape[1], x_center + 320)
    y2 = min(frame.shape[0], y_center + 180)
    masked_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return masked_frame


def detect_game(game: str,
                prediction_dir: Path):
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
    model = YOLO("trained_model/yolov8_814.pt")
    # 動画ライターの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデックの設定
    output = cv2.VideoWriter('data/tutorial/detections/output_video.mp4', fourcc, 25.0, (int(1280), int(720)))  # 出力ファイル、フレームレート、フレームサイズを設定
    
    # 前のフレームのボール位置を初期化
    previous_ball_position = None
    
    with tqdm(total = 2000) as t:
        while n < 2000:
            t.update()
            ret, frame = cap.read()
            if ret:
                ################ 画像変換 ###################
                # グレースケール変換
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # hsv変換
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                ################ 画像を切り取る ###################
                # cut_frame(frame)

                ################ 広角の時のみマスクかける ################
                wide_angle_shot = Wide_Angle_Shot(frame, hsv_frame)
                
                ################ 物体検出 ################
                # YOLOv10で物体検出
                results = model(frame) # frame, center_frame, filtered_frame

                ################ マスク ################
                frame = mask(frame, results)

                ################ 物体追跡 ################
                # tracking(frame, results)

                # 動画にフレームを追加
                output.write(frame)
                n += 1
                # break
            else:
                break
    
    # 動画ファイルを閉じる
    output.release()
    cv2.destroyAllWindows()


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
        detect_game(game, prediction_dir)


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