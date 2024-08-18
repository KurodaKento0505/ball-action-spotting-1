import json
import cv2
import numpy as np

# JSONファイルを読み込む
with open('data/ball_action/predictions/ball_tuning_002/cv/fold_0/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/results_spotting.json', 'r') as f:
    annotations_data = json.load(f)

# 動画のパラメータ
width = 120  # 動画の幅
height = 224  # 動画の高さ
fps = 25  # フレームレート
duration = 183  # 動画の長さ（秒）

# アノテーションを追加するための関数
def get_annotations_at_time(current_frame, annotations):
    annotations_to_display = []
#    for annotation in annotations:
#        annotation_msec = int(annotation["position"])
#        annotation_frame = abs(annotation_msec * fps / 1000)
#        #if current_frame <= annotation_frame and (current_frame - annotation_frame) < fps:  # 真値のフレームから1秒間表示
#        if current_frame == annotation_frame:
#            annotations_to_display.append(annotation)
#    return annotations_to_display

    for annotation in annotations:
        annotation_start_frame = int(int(annotation["position"]) * fps / 1000)
        annotation_end_frame = annotation_start_frame + fps  # 1秒間表示
        if annotation_start_frame <= current_frame < annotation_end_frame:
            annotations_to_display.append(annotation)
    return annotations_to_display


def generate_side_by_side_video(unlabeled_video_path, labeled_video_path, output_video_path):
    # 動画を読み込み
    cap1 = cv2.VideoCapture(unlabeled_video_path)
    cap2 = cv2.VideoCapture(labeled_video_path)

    if not (cap1.isOpened() and cap2.isOpened()):
        raise ValueError("Error opening video files")

    # 動画のプロパティを取得
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 出力動画の設定
    output_width = width1 + width2
    output_height = height1  # 高さは一致していると仮定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

    # 3分間のフレーム数を計算
    max_frames = int(fps * 3 * 60)
    
    frame_count = 0
    while cap1.isOpened() and cap2.isOpened() and frame_count < max_frames:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        # フレームを横に結合
        combined_frame = cv2.hconcat([frame1, frame2])

        # 出力動画にフレームを書き込む
        out.write(combined_frame)
        frame_count += 1

    cap1.release()
    cap2.release()
    out.release()


if __name__ == "__main__":

    # 出力ファイルの設定
    labeled_video_path = 'data/ball_action/predictions/ball_tuning_002/cv/fold_0/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/labeled_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(labeled_video_path, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 白色背景の動画を生成
    for frame_num in range(int(fps * duration)):
        frame = 255 * np.ones((height, width, 3), dtype=np.uint8)
        current_time = int(frame_num / fps)
        current_frame = int(frame_num)
        annotations = get_annotations_at_time(current_frame, annotations_data["predictions"])
        y = 20  # テキストの開始位置
        for annotation in annotations:
            text = f'gameTime: {annotation["gameTime"]}\nlabel: {annotation["label"]}\nposition: {annotation["position"]}\nconfidence: {annotation["confidence"]}'
            for i, line in enumerate(text.split('\n')):
                cv2.putText(frame, line, (20, y + i * 30), font, 0.25, (0, 0, 0), 1, cv2.LINE_AA)
            y += 0  # 次のテキストの位置を下にずらす

        # 経過時間を表示
        minutes = int(current_time // 60)
        seconds = int(current_time % 60)
        time_text = f'{minutes:02}:{seconds:02}'
        cv2.putText(frame, time_text, (width - 30, height - 20), font, 0.25, (0, 0, 0), 1, cv2.LINE_AA)

        out.write(frame)

    out.release()

    # 動画ファイルのパス
    unlabeled_video_path = 'data/soccernet/spotting-ball-2024/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/224p.mp4'
    output_video_path = 'data/ball_action/predictions/ball_tuning_002/cv/fold_0/england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich/results_spotting_video.mp4'

    # 動画を横に並べて新しい動画を生成
    generate_side_by_side_video(unlabeled_video_path, labeled_video_path, output_video_path)