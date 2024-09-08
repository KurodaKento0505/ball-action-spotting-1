import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

import time
import json
import argparse
import multiprocessing
from pathlib import Path
from pprint import pprint
from importlib.machinery import SourceFileLoader
from sklearn.cluster import KMeans
import subprocess

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
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10).fit(image_reshaped)
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
    # edges = cv2.Canny(binary_img, 50, 150)
    return binary_img


# Optical Flowを可視化する関数
def draw_optical_flow(frame, flow, step=16):
    h, w = frame.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    fx, fy = flow[y, x].T
    # 元のフレームをカラー画像に変換
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # ベクトルの終点を計算
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # ベクトルを描画
    for (x1, y1), (x2, y2) in lines:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)  # 緑の線を描画
        cv2.circle(frame, (x1, y1), 1, (0, 255, 0), -1)  # 緑の点を描画
    return frame


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


def stretch_to_square(frame):
    # フレームの高さと幅を取得
    h, w = frame.shape[:2]
    if h >= w:
        print("The frame is already square or taller than wide.")
        return frame
    # 上半分の領域を取得
    top_half = frame[:h//2, :]
    bottom_half = frame[h//2:, :]
    # 上半分を引き伸ばして正方形を作成
    stretched_top_half = cv2.resize(top_half, (w, h//2), interpolation=cv2.INTER_LINEAR)
    # 引き伸ばした領域を新しい正方形フレームにコピー
    square_frame = np.vstack([stretched_top_half, bottom_half])
    # フレームが1280x1280になるように調整
    square_frame = cv2.resize(square_frame, (w, w))
    return square_frame


def stretch_frame(frame):
    # フレームのサイズを取得
    height, width = frame.shape[:2]
    
    # 引き伸ばし対象の560行とそのままの160行を設定
    stretch_height = height - 160  # 560行
    keep_height = 160  # 一番下の160行
    
    # PILのImageに変換
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # セクションの高さを計算
    quarter_height = stretch_height // 4
    
    # 新しい画像のターゲット高さ（1280ピクセル）
    target_height = 1280
    
    # 新しい画像を作成
    new_image = Image.new("RGB", (width, target_height))
    previous_new_y = 0  # 最後に貼り付けた位置を保持
    
    # 一番上の1/4を5/2倍に引き伸ばす
    top_section = image.crop((0, 0, width, quarter_height))
    top_stretched = top_section.resize((width, quarter_height * 5 // 2))
    new_image.paste(top_stretched, (0, previous_new_y))
    previous_new_y += quarter_height * 5 // 2
    
    # 上から二番目の1/4を9/4倍に引き伸ばす
    upper_middle_section = image.crop((0, quarter_height, width, 2 * quarter_height))
    upper_middle_stretched = upper_middle_section.resize((width, quarter_height * 9 // 4))
    new_image.paste(upper_middle_stretched, (0, previous_new_y))
    previous_new_y += quarter_height * 9 // 4
    
    # 下から二番目の1/4を7/4倍に引き伸ばす
    lower_middle_section = image.crop((0, 2 * quarter_height, width, 3 * quarter_height))
    lower_middle_stretched = lower_middle_section.resize((width, quarter_height * 7 // 4))
    new_image.paste(lower_middle_stretched, (0, previous_new_y))
    previous_new_y += quarter_height * 7 // 4
    
    # 一番下の1/4を3/2倍に引き伸ばす
    bottom_section = image.crop((0, 3 * quarter_height, width, 4 * quarter_height))
    bottom_stretched = bottom_section.resize((width, quarter_height * 3 // 2))
    new_image.paste(bottom_stretched, (0, previous_new_y))
    previous_new_y += quarter_height * 3 // 2
    
    # 一番下の160行はそのまま等倍で配置
    final_section = image.crop((0, stretch_height, width, height))
    new_image.paste(final_section, (0, previous_new_y))
    
    # OpenCV形式に戻して返す
    return cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)


def add_black_to_square(frame):
    # フレームの高さと幅を取得
    h, w = frame.shape[:2]
    if h >= w:
        print("The frame is already square or taller than wide.")
        return frame
    # 正方形の高さを計算
    square_size = w
    # 上部を黒で塗りつぶす
    black_top = np.zeros((square_size - h, w, 3), dtype=np.uint8)
    # 黒い領域を上に追加して正方形を作成
    square_frame = np.vstack([black_top, frame])
    # フレームが1280x1280になるように調整
    square_frame = cv2.resize(square_frame, (w, w))
    return square_frame


def convert_to_h264(input_file, output_file):
    # ffmpegコマンドを使用して、mpeg4コーデックをh264に変換
    cmd = [
        "ffmpeg", "-i", str(input_file), "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", str(output_file)
    ]

    # コマンドを実行
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 結果を確認
    if result.returncode == 0:
        print(f"Successfully converted {input_file} to {output_file} with h264 codec.")
    else:
        print(f"Failed to convert {input_file}. Error: {result.stderr}")


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
    output_path = game_dir / f"square_video.{constants.videos_extension}"
    print("output_path:", output_path)
    # output_path = 'data/tutorial/estimate_camera_movement/output_video.mp4'
    # 一時的な動画ファイルパス
    temp_output_path = game_dir / f"square_video_temp.mp4"
    # temp_output_path = 'data/tutorial/estimate_camera_movement/output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output = cv2.VideoWriter(str(temp_output_path), fourcc, 25.0, (1280, 1280))

    # 背景差分用のオブジェクトを作成
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    
    # 最初のフレームを取得してグレースケールに変換
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_binary_frame = None
    prev_wide_angle_shot = False
    movement_threshold = 10
    stretched_height = 1280  # 引き延ばしたいサイズを指定

    n = 0
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as t: # total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT
        while True: # n<1000, True
            t.update()
            ret, frame = cap.read()
            if not ret:
                break

            ########################## 画像変換 ##########################
            # 現在のフレームをグレースケールに変換
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # hsv変換
            # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # binary_frame = binary_kmeans(frame)

            ################ 広角か否かの判断 ################
            # wide_angle_shot = Wide_Angle_Shot_with_Green(frame, hsv_frame)
            if n % 15 == 0:
                wide_angle_shot = Wide_Angle_Shot_with_Player_Bbox(frame)
                prev_wide_angle_shot = wide_angle_shot
            else:
                wide_angle_shot = prev_wide_angle_shot

            ################ 広角 ################
            if wide_angle_shot:

                ########################## フレームを引き延ばす #####################################
                # 上半分のみ
                # frame = stretch_to_square(frame)

                # 全体的に
                frame = stretch_frame(frame)

                ########################## k-meansで5つのクラスタから二値化してエッジ検出 ##########################
                # binary_frame = binary_kmeans(frame)

                '''########################## 二値化された画像からオプティカルフローでカメラの動きを検出 ##########################
                if prev_binary_frame is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_binary_frame, binary_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    prev_binary_frame = binary_frame
                    # オプティカルフローを可視化
                    binary_frame = draw_optical_flow(prev_binary_frame, flow)
                    # オプティカルフローの結果からカメラの動きを計算（例：水平方向と垂直方向の移動量を計算）
                    flow_mean = np.mean(flow, axis=(0, 1))
                    camera_shift_x = flow_mean[0]
                    camera_shift_y = flow_mean[1]
                    # 結果を表示（または、他の方法でカメラの動きを利用）
                    print(f"Frame {n}: Camera shift (x, y) = ({camera_shift_x:.2f}, {camera_shift_y:.2f})")
                else:
                    prev_binary_frame = binary_frame'''
                
                '''########################## オプティカルフローで移動量多い物体を検出 ##################################
                # オプティカルフローを計算
                if n == 0:
                    prev_gray = binary_frame
                    n += 1
                    continue
                flow = cv2.calcOpticalFlowFarneback(prev_gray, binary_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                prev_gray = binary_frame
                # 移動量を計算
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                # 移動量が閾値を超える部分を抽出
                movement_mask = magnitude > movement_threshold
                # 移動量が大きい領域を強調表示
                movement_frame = frame.copy()
                movement_frame[movement_mask] = [0, 0, 255]  # 赤色で強調表示'''
                
                # 結果を動画として保存
                output.write(frame) # movement_, binary_
            
            ################ 広角じゃない ################
            else:
                frame = add_black_to_square(frame)
                output.write(frame)
                # prev_binary_frame = None

            ########################## 線検出 ##########################
            '''detect_line(frame, hsv_frame)'''

            ########################## 特徴点検出 ##########################
            # p0 = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

            # 次のフレームのために現在のフレームを保存
            # prev_gray = gray.copy()

            n += 1

    cap.release()
    output.release()
    cv2.destroyAllWindows()

    # H.264形式に変換
    output_path = game_dir / f"propotional_square_video.mp4"
    convert_to_h264(temp_output_path, output_path)

    # 一時ファイルの削除（必要に応じて）
    temp_output_path.unlink()


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