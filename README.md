# 手のキーポイント推定システム

Ultralytics YOLO11を使用した手のキーポイント推定モデルの学習とウェブカメラでのリアルタイム推定を行うシステムです。

## 機能

- 手のキーポイント推定モデルの学習
- ウェブカメラでのリアルタイム手のキーポイント検出
- ジェスチャー認識と手話翻訳の基盤
- AR/VRアプリケーション向けのハンドトラッキング

## インストール

```bash
pip install -r requirements.txt
```

# training
python train_hand_keypoints.py --config configs/hand_keypoints.yaml   

# detection
python webcam_inference.py --model runs/pose/hand_keypoints/weights/best.pt


## 使用方法

### 1. モデルの学習

```bash
python train_hand_keypoints.py
```

### 2. ウェブカメラでの推定

```bash
python webcam_inference.py
```

### 3. 画像での推定

```bash
python image_inference.py --image path/to/image.jpg
```

## プロジェクト構成

```
u_hand_keypoint/
├── requirements.txt          # 依存関係
├── train_hand_keypoints.py  # モデル学習スクリプト
├── webcam_inference.py      # ウェブカメラ推定
├── image_inference.py       # 画像推定
├── data/                    # データセット
├── models/                  # 学習済みモデル
└── configs/                 # 設定ファイル
```

## 参考

- [Ultralytics YOLO11で手のキーポイント推定を強化](https://www.ultralytics.com/ja/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11)
