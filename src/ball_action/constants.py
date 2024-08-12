from src.constants import data_dir, soccernet_dir, configs_dir

ball_action_dir = data_dir / "ball_action"
configs_dir = configs_dir / "ball_action"
experiments_dir = ball_action_dir / "experiments"
predictions_dir = ball_action_dir / "predictions"
visualizations_dir = ball_action_dir / "visualizations"

soccernet_dir = soccernet_dir / "spotting-ball-2024"

fold_games = [
    "england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich",
    "england_efl/2019-2020/2019-10-01 - Hull City - Sheffield Wednesday",
    "england_efl/2019-2020/2019-10-01 - Brentford - Bristol City",
    "england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest",
    "england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End",
    "england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town",
    "england_efl/2019-2020/2019-10-01 - Reading - Fulham",
]
fold2games = {fold: [game] for fold, game in enumerate(fold_games)}
game2fold = {game: fold for fold, games in fold2games.items() for game in games}
folds = sorted(fold2games.keys())
challenge_games = [
    "england_efl/2019-2020/2019-10-02 - Cardiff City - Queens Park Rangers",
    "england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City",
]

# classes = [
#     "PASS",
#     "DRIVE",
# ]

labels_filename = "Labels-ball.json"
videos_extension = 'mp4'

classes = ['PASS', 'DRIVE', 'HEADER', 'HIGH PASS', 'OUT', 'CROSS', 'THROW IN',
           'SHOT', 'BALL PLAYER BLOCK', 'PLAYER SUCCESSFUL TACKLE', 'FREE KICK',
           'GOAL']

num_classes = len(classes)
target2class: dict[int, str] = {trg: cls for trg, cls in enumerate(classes)}
class2target: dict[str, int] = {cls: trg for trg, cls in enumerate(classes)}

num_halves = 1
halves = list(range(1, num_halves + 1))
# ここで閾値を決定，ピークがheightを超えたときにその時刻を開始時刻？
# クラスごとに変更する
postprocess_params = {
    "gauss_sigma": 3.0,
    "height": 0.2,
    "distance": 15,
}
'''postprocess_params = [
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 15},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 40},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 15},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 15},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 250},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 15},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 15},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 15},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 15},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 15},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 15},
    {"gauss_sigma": 3.0, "height": 0.2, "distance": 15}
]'''

video_fps = 25.0
