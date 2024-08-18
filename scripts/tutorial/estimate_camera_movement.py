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
from sklearn.cluster import KMeans

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


RESOLUTION = "720p"
TTA = True


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, type=str)
    parser.add_argument("--folds", default="all", type=str)
    return parser.parse_args()


def draw_line(img, theta, rho):
    h, w = img.shape[:2]
    # 直線が垂直のとき
    if np.isclose(np.sin(theta), 0):
        x1, y1 = rho, 0
        x2, y2 = rho, h
    # 直線が垂直じゃないとき
    else:
        # 直線の式を式変形すればcalc_yの式がもとまる(解説を後述します)．
        calc_y = lambda x: rho / np.sin(theta) - x * np.cos(theta) / np.sin(theta)
        x1, y1 = 0, calc_y(0)
        x2, y2 = w, calc_y(w)
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    # 直線を描画
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img


def binary_kmeans(frame):
    # 画像の形状を取得
    h, w, c = frame.shape
    # 画像を2次元配列に変換 (k-meansの入力に使用)
    image_reshaped = frame.reshape(-1, 3)
    # k-meansで5つのクラスタに分割
    kmeans = KMeans(n_clusters=5, random_state=0).fit(image_reshaped)
    labels = kmeans.labels_
    # 5つのクラスタをGチャンネルのみで表現
    clustered_img = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            clustered_img[i, j] = kmeans.cluster_centers_[labels[i*w + j]][1]  # Gチャンネルの値を使用
    # 白色の範囲を定義
    white_threshold = 200  # 白とみなすGチャンネルのしきい値
    # 白色の箇所を抽出（二値化）
    binary_img = np.zeros((h, w), dtype=np.uint8)
    binary_img[clustered_img > white_threshold] = 255
    # エッジ検出 (Canny)
    edges = cv2.Canny(binary_img, 50, 150)


def Wide_Angle_Shot_with_Green(frame, hsv_frame):
    # フレームの50%以上が緑色なら広角と判断
    green_threshold = 0.4
    # 緑色の範囲を定義 (この範囲は調整が必要です)
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([50, 255, 255])
    # 緑色のマスクを作成
    gr_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    # 緑色の割合を計算
    green_ratio = cv2.countNonZero(gr_mask) / (frame.shape[0] * frame.shape[1])
    # 緑色の割合 (と特徴点数) が閾値を超える場合、広角ショットと判断
    wide_angle_shot = False
    if green_ratio > green_threshold: # and p0 is not Noneand len(p0) > matching_threshold:
        wide_angle_shot = True
    return wide_angle_shot


def Wide_Angle_Shot_with_Player_Bbox(frame):
    # YOLOv10
    model = YOLO("trained_model/yolov10n.pt")
    results = model(frame)
    boxes = results[0].boxes  # 検出結果を取得
    # 閾値（ピクセル面積の閾値）を設定
    bbox_threshold = 20000  # 適切な値に調整
    # 検出された人の中で最もコンフィデンスが高い物体を取得
    if len(boxes) > 0:
        # コンフィデンスが最も高いボックスを選択
        max_confidence_box = max(boxes, key=lambda box: box.conf)
        # バウンディングボックスの座標を取得
        x1, y1, x2, y2 = max_confidence_box.xyxy[0]
        # バウンディングボックスの面積を計算
        max_area = (x2 - x1) * (y2 - y1)
        # 最大のバウンディングボックスの面積が閾値以下の場合、広角ショットと判断
        wide_angle_shot = max_area < bbox_threshold
    else:
        # 検出された人がいない場合、広角ショットではないと判断
        wide_angle_shot = True
        max_area = 0
    print(max_area)
    return wide_angle_shot


def detect_line(frame, hsv_frame):
    # 緑色の範囲を定義 (この範囲は調整が必要です)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([40, 255, 255])
    # 緑色のマスクを作成
    gr_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    gr_mask = cv2.bitwise_not(gr_mask)
    masked_frame = cv2.bitwise_and(frame, frame, mask=gr_mask)

    masked_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    # 横方向フィルタ
    dx = cv2.Sobel(masked_gray, cv2.CV_64F, 1, 0, ksize=3)
    # Laplacianフィルターを適用
    laplacian = cv2.Laplacian(masked_frame, cv2.CV_64F)
    # エッジを絶対値に変換して明るく
    abs = cv2.convertScaleAbs(laplacian)
    # エッジを太く強調
    kernel = np.ones((5, 5), np.uint8)  # 太さを決定するカーネルサイズ
    dilated_edges = cv2.dilate(abs, kernel, iterations=1)
    # 明るさを調整してエッジをさらに強調
    enhanced_edges = cv2.convertScaleAbs(dilated_edges, alpha=2, beta=50)  # alphaとbetaは明るさとコントラストの調整
    # im_edges = cv2.Canny(masked_frame, 100, 200, L2gradient=True)
    cv2.imwrite('data/tutorial/estimate_camera_movement/edge.png', enhanced_edges)
    im_lines = cv2.HoughLines(enhanced_edges, rho=1, theta=np.pi/180, threshold=20)
    for line in im_lines:
        rho, theta = line[0]
        result = draw_line(frame, theta, rho)
    cv2.imwrite('data/tutorial/estimate_camera_movement/red_line.png', result)


def estimate_camera_movement(game: str, prediction_dir: Path):
    game_dir = constants.soccernet_dir / game
    game_prediction_dir = prediction_dir / game
    game_prediction_dir.mkdir(parents=True, exist_ok=True)
    print("Detect game:", game)
    video_path = game_dir / f"{RESOLUTION}.{constants.videos_extension}" #f"{half}_{RESOLUTION}.mkv"
    video_info = get_video_info(video_path)
    print("Video info:", video_info)
    print("Detect video:", video_path)

    # 動画を読み込む
    cap = cv2.VideoCapture(str(video_path))

    # 動画の書き出し設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter('data/tutorial/estimate_camera_movement/output_video.mp4', fourcc, 25.0, (1280, 720))

    # 背景差分用のオブジェクトを作成
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    
    # 最初のフレームを取得してグレースケールに変換
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    n = 0
    with tqdm(total = 2000) as t:
        while n < 2000:
            t.update()
            ret, frame = cap.read()
            if not ret:
                break

            ########################## 画像変換 ##########################
            # 現在のフレームをグレースケールに変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # hsv変換
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            ########################## k-meansで5つのクラスタから二値化してエッジ検出 ##########################
            '''edges = binary_kmeans(frame)
            # 結果を動画として保存
            output.write(edges)'''

            ########################## 緑の量から wide angle shot を推定 ##########################
            wide_angle_shot = Wide_Angle_Shot_with_Green(frame, hsv_frame)

            ########################## 線検出 ##########################
            '''detect_line(frame, hsv_frame)'''

            ########################## 特徴点検出 ##########################
            # p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

            # 結果を動画として保存
            output.write(frame)

            # 次のフレームのために現在のフレームを保存
            # prev_gray = gray.copy()

            n += 1

    cap.release()
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
        estimate_camera_movement(game, prediction_dir)


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