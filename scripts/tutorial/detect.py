import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction
from collections import deque

import time
import json
import argparse
import multiprocessing
from pathlib import Path
from pprint import pprint
from importlib.machinery import SourceFileLoader
# from deep_sort_realtime.deepsort_tracker import DeepSort

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

from estimate_camera_movement import Wide_Angle_Shot_with_Green
from estimate_camera_movement import Wide_Angle_Shot_with_Player_Bbox


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
def cut_frame(frame, results, prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2):
    boxes = results[0].boxes  # 検出結果を取得
    # フレームを黒くする
    # confidenceが最も高いボールを選択
    if boxes:
        best_box = max(boxes, key=lambda x: x.conf)
        x_min, y_min, x_max, y_max = best_box.xyxy[0].tolist()  # 座標を取得
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        confidence = best_box.conf.item()
        # confidence >= 0.7
        if confidence >= 0.8:
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            x1 = int(max(0, x_center - 320))
            y1 = int(max(0, y_center - 180))
            x2 = int(min(frame.shape[1], x_center + 320))
            y2 = int(min(frame.shape[0], y_center + 180))
            cut_frame = frame[y1:y2, x1:x2]
            prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = x1, y1, x2, y2
        # confidence >= 0.7
        elif confidence >= 0.6:
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            x1 = int(max(0, x_center - 320))
            y1 = int(max(0, y_center - 180))
            x2 = int(min(frame.shape[1], x_center + 320))
            y2 = int(min(frame.shape[0], y_center + 180))
            # 前のフレームのマスク情報あり
            if prev_masked_x1 != -1:
                # 重なり領域を計算
                intersection_x1 = max(prev_masked_x1, x1)
                intersection_y1 = max(prev_masked_y1, y1)
                intersection_x2 = min(prev_masked_x2, x2)
                intersection_y2 = min(prev_masked_y2, y2)
                intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
                
                prev_mask_area = (prev_masked_x2 - prev_masked_x1) * (prev_masked_y2 - prev_masked_y1)
                overlap_ratio = intersection_area / prev_mask_area
                # 前のフレームとの重なりが50％以上，現在の認識結果の使用
                if overlap_ratio >= 0.3:
                    cut_frame = frame[y1:y2, x1:x2]
                    prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = x1, y1, x2, y2
                # 前のフレームとの重なりが50％未満，前のフレームのまま
                else:
                    cut_frame = frame[prev_masked_y1:prev_masked_y2, prev_masked_x1:prev_masked_x2]

            # 前のフレームのマスク情報なし
            else:
                cut_frame = frame[y1:y2, x1:x2]
                prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = x1, y1, x2, y2
        
        # confidence >= 0.7
        elif confidence >= 0.3:
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            x1 = int(max(0, x_center - 320))
            y1 = int(max(0, y_center - 180))
            x2 = int(min(frame.shape[1], x_center + 320))
            y2 = int(min(frame.shape[0], y_center + 180))
            # 前のフレームのマスク情報あり
            if prev_masked_x1 != -1:
                # 重なり領域を計算
                intersection_x1 = max(prev_masked_x1, x1)
                intersection_y1 = max(prev_masked_y1, y1)
                intersection_x2 = min(prev_masked_x2, x2)
                intersection_y2 = min(prev_masked_y2, y2)
                intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
                
                prev_mask_area = (prev_masked_x2 - prev_masked_x1) * (prev_masked_y2 - prev_masked_y1)
                overlap_ratio = intersection_area / prev_mask_area
                # 前のフレームとの重なりが50％以上，現在の認識結果の使用
                if overlap_ratio >= 0.5:
                    cut_frame = frame[y1:y2, x1:x2]
                    prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = x1, y1, x2, y2
                # 前のフレームとの重なりが50％未満，前のフレームのまま
                else:
                    cut_frame = frame[prev_masked_y1:prev_masked_y2, prev_masked_x1:prev_masked_x2]

            # 前のフレームのマスク情報なし
            else:
                cut_frame = frame
                # prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = x1, y1, x2, y2

        # confidence < 0.7，前のフレームのまま
        else:
            # 前のフレームのマスク情報あり
            if prev_masked_x1 != -1:
                cut_frame = frame[prev_masked_y1:prev_masked_y2, prev_masked_x1:prev_masked_x2]
            # 前のフレームのマスク情報なし
            else:
                cut_frame = frame
    # 認識できなかった
    else:
        # 前のフレームのマスク情報あり
        if prev_masked_x1 != -1:
            cut_frame = frame[prev_masked_y1:prev_masked_y2, prev_masked_x1:prev_masked_x2]
        # 前のフレームのマスク情報なし
        else:
            cut_frame = frame
    return cut_frame, prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2


def draw_bbox(frame, results, cls_name = None, Track = False):
    if Track:
        # トラッキング結果を描画
        for track in results:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x_min, y_min, x_max, y_max = map(int, ltrb)
            # バウンディングボックスを描画
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # ラベルとIDを描画
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        boxes = results[0].boxes  # 検出結果を取得
        if boxes:
            # confidenceが最も高いボールを選択
            best_box = max(boxes, key=lambda x: x.conf)
            x_min, y_min, x_max, y_max = best_box.xyxy[0].tolist()  # 座標を取得
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            # yoloの結果を描画
            confidence = best_box.conf.item()  # 信頼度を取得
            # バウンディングボックスを描画
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            # ラベルと信頼度を描画
            label = f"{cls_name} {confidence:.2f}" if cls_name else f"Conf: {confidence:.2f}"
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def draw_bbox_SAHI(frame, results_ball):
    for object_prediction in results_ball.object_prediction_list:
        x_min, y_min, x_max, y_max = map(int, object_prediction.bbox.to_voc_bbox())
        confidence = object_prediction.score.value
        label = f"ball {confidence:.2f}"
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def results_to_json(results_ball, n, ball_detections):
    boxes = results_ball[0].boxes  # 検出結果を取得
    if boxes:
        # confidenceが最も高いボールを選択
        best_box = max(boxes, key=lambda x: x.conf)
        confidence = best_box.conf.item()
        # コンフィデンスが0.8以上のボールのバウンディングボックスを記録
        if confidence >= 0.8:
            x_min, y_min, x_max, y_max = best_box.xyxy[0].tolist()  # 座標を取得
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            ball_detections.append({
                    'frame_number': n,
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'confidence': confidence
                })


# ボール周辺以外にマスクをかける
def mask_frame(frame, results, prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2):
    boxes = results[0].boxes  # 検出結果を取得
    # フレームを黒くする
    masked_frame = np.zeros_like(frame)
    # confidenceが最も高いボールを選択
    if boxes:
        best_box = max(boxes, key=lambda x: x.conf)
        x_min, y_min, x_max, y_max = best_box.xyxy[0].tolist()  # 座標を取得
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        confidence = best_box.conf.item()
        # ボールを検出して誤検出を省く
        # confidence < 0.4 は信用しない
        if confidence >= 0.8:
            x_center = int((x_min + x_max) / 2)
            y_center = int((y_min + y_max) / 2)
            x1 = int(max(0, x_center - 360))
            y1 = int(max(0, y_center - 180))
            x2 = int(min(frame.shape[1], x_center + 360))
            y2 = int(min(frame.shape[0], y_center + 180))
            masked_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
            prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = x1, y1, x2, y2
            # ボールとプレーヤーの検出結果を描画
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # ボールのバウンディングボックスを緑で描画
            cv2.putText(frame, f'Ball: {confidence:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # confidence >= 0.4
        else:
            '''# 前のフレームのマスク情報あり
            if prev_masked_x1 != -1:
                # 重なり領域を計算
                intersection_x1 = max(prev_masked_x1, x1)
                intersection_y1 = max(prev_masked_y1, y1)
                intersection_x2 = min(prev_masked_x2, x2)
                intersection_y2 = min(prev_masked_y2, y2)
                intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
                
                prev_mask_area = (prev_masked_x2 - prev_masked_x1) * (prev_masked_y2 - prev_masked_y1)
                overlap_ratio = intersection_area / prev_mask_area
                # 前のフレームとの重なりが50％以上，現在の認識結果の使用
                if overlap_ratio >= 0.3:
                    masked_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                    prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = x1, y1, x2, y2
                # 前のフレームとの重なりが50％未満，前のフレームのまま
                else:
                    masked_frame[prev_masked_y1:prev_masked_y2, prev_masked_x1:prev_masked_x2] = frame[prev_masked_y1:prev_masked_y2, prev_masked_x1:prev_masked_x2]
            # 前のフレームのマスク情報なし
            else:
                masked_frame = frame
                # masked_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
                # prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = x1, y1, x2, y2'''
            masked_frame = frame

    # 認識できなかった
    else:
        '''# 前のフレームのマスク情報あり
        if prev_masked_x1 != -1:
            masked_frame[prev_masked_y1:prev_masked_y2, prev_masked_x1:prev_masked_x2] = frame[prev_masked_y1:prev_masked_y2, prev_masked_x1:prev_masked_x2]
        # 前のフレームのマスク情報なし
        else:
            masked_frame = frame'''
        masked_frame = frame
    return masked_frame, prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2


def mask_frame_with_confidence(frame, prev_ball_dict, next_ball_dict, n):
    # フレームを黒くする
    masked_frame = np.zeros_like(frame)
    # フレーム番号を保存
    prev_frame_num = prev_ball_dict['frame_number']
    next_frame_num = next_ball_dict['frame_number']
    central_frame_num = int((prev_frame_num + next_frame_num) / 2)
    # prev と next の座標を保存
    prev_x_center = int((prev_ball_dict['x_min'] + prev_ball_dict['x_max']) / 2)
    prev_y_center = int((prev_ball_dict['y_min'] + prev_ball_dict['y_max']) / 2)
    next_x_center = int((next_ball_dict['x_min'] + next_ball_dict['x_max']) / 2)
    next_y_center = int((next_ball_dict['y_min'] + next_ball_dict['y_max']) / 2)
    # 今のボールの位置を予測
    n_x_center = ((n - prev_frame_num) * prev_x_center + (next_frame_num - n) * next_x_center) / (next_frame_num - prev_frame_num)
    n_y_center = ((n - prev_frame_num) * prev_y_center + (next_frame_num - n) * next_y_center) / (next_frame_num - prev_frame_num)
    # confidence > 0.8 の時のマスクの大きさ
    default_size_x = 20
    default_size_y = 20
    # 1フレームあたりに拡張する大きさ
    expand_x_per_frame = 1
    expand_y_per_frame = 1
    # 前半
    if n < central_frame_num:
        n_x_min = int(max(0, n_x_center - (default_size_x / 2 + (n - prev_frame_num) * expand_x_per_frame)))
        n_y_min = int(max(0, n_y_center - (default_size_y / 2 + (n - prev_frame_num) * expand_y_per_frame)))
        n_x_max = int(min(frame.shape[1], n_x_center + (default_size_x / 2 + (n - prev_frame_num) * expand_x_per_frame)))
        n_y_max = int(min(frame.shape[0], n_y_center + (default_size_y / 2 + (n - prev_frame_num) * expand_y_per_frame)))
        masked_frame[n_y_min:n_y_max, n_x_min:n_x_max] = frame[n_y_min:n_y_max, n_x_min:n_x_max]
    # 後半
    else:
        n_x_min = int(max(0, n_x_center - (default_size_x / 2 + (next_frame_num - n) * expand_x_per_frame)))
        n_y_min = int(max(0, n_y_center - (default_size_y / 2 + (next_frame_num - n) * expand_y_per_frame)))
        n_x_max = int(min(frame.shape[1], n_x_center + (default_size_x / 2 + (next_frame_num - n) * expand_x_per_frame)))
        n_y_max = int(min(frame.shape[0], n_y_center + (default_size_y / 2 + (next_frame_num - n) * expand_y_per_frame)))
        masked_frame[n_y_min:n_y_max, n_x_min:n_x_max] = frame[n_y_min:n_y_max, n_x_min:n_x_max]
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
    ######################### change video path ############################
    # video_path = 'data/tutorial/estimate_camera_movement/output_video.mp4'
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return

    # 動画ライターの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデックの設定
    # output_path = game_dir / f"masked_video.{constants.videos_extension}"
    output_path = "data/tutorial/detections/output_video.mp4"
    output = cv2.VideoWriter(str(output_path), fourcc, int(25), (int(1280), int(1280)))  # 1280, 720, 出力ファイル、フレームレート、フレームサイズを設定

    # 何のループ回すか
    easy_mask = True

    ############################### 物体検出，マスクするためのループ ###############################
    if easy_mask:
        n = 1
        # YOLOv10モデルの読み込み
        model_ball = YOLO("trained_model/yolov10_816.pt")
        model_person = YOLO('trained_model/yolov8n.pt')
        '''detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path="trained_model/yolov10_816.pt",
            confidence_threshold=0.3,
            device="cuda:0",
        )'''
        # 前のフレームのボール位置を初期化
        prev_frame = None
        prev_wide_angle_shot = False
        prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = -1, -1, -1, -1
        # 結果を保存するリストs
        # ball_detections = []
        wide_angle_changes = []

        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as t: # total=1000
            while True: # n < 1000:
                t.update()
                ret, frame = cap.read()
                if ret:
                    ################ 画像変換 ###################
                    # グレースケール変換
                    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # hsv変換
                    # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    ################ 広角か否かの判断 ################
                    # wide_angle_shot = Wide_Angle_Shot_with_Green(frame, hsv_frame)
                    if n % 15 == 0:
                        wide_angle_shot = Wide_Angle_Shot_with_Player_Bbox(frame)
                        # prev_wide_angle_shot = wide_angle_shot
                    else:
                        wide_angle_shot = prev_wide_angle_shot
                    
                    '''# 広角ショットの切り替えを記録
                    if wide_angle_shot != prev_wide_angle_shot:
                        if wide_angle_shot:
                            binary_wide_angle_shot = 0
                        else:
                            binary_wide_angle_shot = 1
                        wide_angle_changes.append({
                            "frame_number": n,
                            "wide_angle": binary_wide_angle_shot
                        })
                    prev_wide_angle_shot = wide_angle_shot'''

                    ################ 広角 ################
                    if wide_angle_shot:

                        ############### 画像を切り取る ###################
                        # cut_frame(frame)

                        ################ 物体検出 ################
                        # YOLOv10で物体検出
                        results_ball = model_ball(frame) # frame, center_frame, filtered_frame
                        draw_bbox(frame, results_ball)
                        # SAHI
                        '''results_ball = get_sliced_prediction(
                            frame,
                            detection_model,
                            slice_height=200,
                            slice_width=200,
                            overlap_height_ratio=0.2,
                            overlap_width_ratio=0.2,
                        )
                        draw_bbox_SAHI(frame, results_ball)'''
                        # 検出結果をjsonで保存
                        # results_to_json(results_ball, n, ball_detections)
                        # jsonから読み取る

                        ################ マスクかける ################
                        # frame, prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = mask_frame(frame, results_ball, prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2)
                        # cut
                        # frame, prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = cut_frame(frame, results_ball, prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2)
                        # マスクした後に物体検出
                        # results_ball = model_ball(frame)
                        # results_person = model_person.track(frame)
                        # draw_bbox(frame, results_ball, 'ball')
                        # draw_bbox(frame, results_person, Track=True)
                        
                    ################ 広角じゃない ################
                    else:
                        prev_masked_x1, prev_masked_y1, prev_masked_x2, prev_masked_y2 = -1, -1, -1, -1
                    
                    # draw_bbox(frame, results_ball)
                    ################ 物体追跡 ################
                    # tracking(frame, results)

                    # 動画にフレームを追加
                    output.write(frame)
                    n += 1
                    # break
                else:
                    break
        
        '''with open(game_dir / f"wide_angle_changes.json", 'w') as f:
            json.dump(wide_angle_changes, f, indent=4)'''

        # 動画ファイルを閉じる
        cap.release()
        output.release()
        cv2.destroyAllWindows()
    
    ############################### 0.8以上のbboxを頼りにマスクかける ###############################
    else:
        # ball_detectionのファイルとwide_angle_changeのファイルを参照
        json_open_ball = open(game_dir / f"ball_detection.json", 'r')
        ball_detection_list = json.load(json_open_ball)
        json_open_wide_angle = open(game_dir / f"wide_angle_changes.json", 'r')
        wide_angle_list = json.load(json_open_wide_angle)
        # ループ
        n = 0
        num_ball_detection = 0
        num_wide_angle = 0
        prev_ball_dict = {}
        with tqdm(total=1000) as t: # total=1000, total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            while n < 1000: # n < 1000, True
                t.update()
                ret, frame = cap.read()
                if ret:
                    # confidence >= 0.8でボールを検出したフレーム
                    if n == ball_detection_list[num_ball_detection]['frame_number']:
                        prev_ball_dict = ball_detection_list[num_ball_detection]
                        # 次のconfidence >= 0.8でボールを検出したフレームと広角じゃなくなるフレームのどちらが先に訪れるか
                        # 先にconfidence >= 0.8でボールを検出したフレームが来る場合
                        if wide_angle_list[2 * num_wide_angle + 1]['frame_number'] > ball_detection_list[num_ball_detection + 1]['frame_number']:
                            next_ball_dict = ball_detection_list[num_ball_detection + 1]
                        # 先に広角じゃなくなるフレームが来る場合
                        else:
                            prev_ball_dict = {}
                            num_wide_angle += 1
                        num_ball_detection += 1
                    
                    # prev_ball_dictが空かどうか，空＝一回画角変わる
                    # prev_ball_dictが空じゃない
                    if any(prev_ball_dict) == True:
                        # マスクかける
                        frame = mask_frame_with_confidence(frame, prev_ball_dict, next_ball_dict, n)
                    # 動画にフレームを追加
                    output.write(frame)
                    n += 1
                else:
                    break
        # 動画ファイルを閉じる
        cap.release()
        output.release()


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