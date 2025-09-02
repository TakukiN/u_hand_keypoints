# 手のキーポイント推定システム 使用方法

このプロジェクトは、Ultralytics YOLO11を使用して手のキーポイント推定を行うシステムです。記事「[Ultralytics YOLO11で手のキーポイント推定を強化](https://www.ultralytics.com/ja/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11)」の内容に基づいて構築されています。

## システム概要

### 機能
- **手のキーポイント推定モデルの学習**: YOLO11を使用したカスタムモデルの学習
- **ウェブカメラでのリアルタイム推定**: リアルタイムでの手のキーポイント検出とジェスチャー認識
- **画像での推定**: 静止画像からの手のキーポイント検出
- **データセット作成**: 学習用データセットの作成と管理

### 手のキーポイント（21個）
1. **手首** (wrist)
2. **親指**: 先端、第2関節、第1関節、付け根
3. **人差指**: 先端、第2関節、第1関節、付け根
4. **中指**: 先端、第2関節、第1関節、付け根
5. **薬指**: 先端、第2関節、第1関節、付け根
6. **小指**: 先端、第2関節、第1関節、付け根
7. **手のひら中心** (palm_center)

## インストール

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 2. 事前学習済みモデルのダウンロード（オプション）
```bash
# YOLO11ポーズ推定モデルを自動ダウンロード
python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt')"
```

## 使用方法

### 1. モデルの学習

#### 基本的な学習
```bash
python train_hand_keypoints.py
```

#### 学習の流れ
1. データセットディレクトリ構造の作成
2. 設定ファイルの生成
3. モデルの学習実行
4. 結果の保存

#### 学習設定のカスタマイズ
- `configs/train.yaml` ファイルを編集して学習パラメータを調整
- エポック数、バッチサイズ、学習率などを変更可能

### 2. ウェブカメラでのリアルタイム推定

#### 基本的な実行
```bash
python webcam_inference.py
```

#### オプション付き実行
```bash
# カスタムモデルを使用
python webcam_inference.py --model ./runs/hand_keypoints/weights/best.pt

# カメラIDを指定
python webcam_inference.py --camera 1

# 信頼度閾値を調整
python webcam_inference.py --conf 0.7

# 使用デバイスを指定
python webcam_inference.py --device cuda
```

#### ウェブカメラの操作
- **'q'**: 終了
- **'s'**: スクリーンショット保存
- **'c'**: 信頼度閾値変更

#### 表示される情報
- 検出されたジェスチャー
- 信頼度閾値
- FPS（フレームレート）
- キーポイントと骨格の可視化

### 3. 画像での推定

#### 基本的な実行
```bash
python image_inference.py --image path/to/image.jpg
```

#### オプション付き実行
```bash
# 出力パスを指定
python image_inference.py --image input.jpg --output result.jpg

# 信頼度閾値を調整
python image_inference.py --image input.jpg --conf 0.6

# 結果をJSONファイルとして保存
python image_inference.py --image input.jpg --save-json

# matplotlibで可視化
python image_inference.py --image input.jpg --visualize
```

#### 出力ファイル
- キーポイントが描画された画像
- 検出結果のJSONファイル（オプション）
- 可視化結果のPNGファイル（オプション）

### 4. データセット作成

#### サンプルデータセットの作成
```bash
python create_dataset.py --create-sample
```

#### アノテーションファイルからの変換
```bash
python create_dataset.py --annotation-file annotations.json
```

#### データセットの検証
```bash
python create_dataset.py --validate
```

#### カスタム分割比率の設定
```bash
python create_dataset.py --annotation-file annotations.json --split-ratio 0.8 0.15 0.05
```

## データセット形式

### 入力形式
- **画像**: JPG、PNG形式
- **アノテーション**: JSON形式
  ```json
  [
    {
      "image_path": "path/to/image.jpg",
      "keypoints": [
        [x1, y1, confidence1],
        [x2, y2, confidence2],
        ...
      ]
    }
  ]
  ```

### 出力形式
- **YOLO形式**: 各画像に対応する.txtファイル
- **形式**: `class_id x_center y_center width height kp1_x kp1_y kp1_conf ...`

## ジェスチャー認識

### 認識可能なジェスチャー
- **OK Sign**: 親指と人差指が近い
- **Open Hand**: 親指と人差指が遠い
- **Pointing (1)**: 1本の指が伸びている
- **Peace Sign (2)**: 2本の指が伸びている
- **Three (3)**: 3本の指が伸びている
- **Four (4)**: 4本の指が伸びている
- **Open Hand (5)**: 5本の指が伸びている

### ジェスチャー認識の仕組み
1. キーポイントの信頼度チェック
2. 指の伸び具合の判定（手首との位置関係）
3. 指先間の距離計算
4. パターンマッチングによる分類

## パフォーマンス最適化

### GPU使用
```bash
# CUDAデバイスを明示的に指定
python webcam_inference.py --device cuda:0
```

### 処理速度向上
- 入力画像サイズの調整（`--imgsz`）
- バッチサイズの最適化
- 信頼度閾値の調整

### メモリ使用量削減
- 小さなモデルサイズの選択（yolo11n-pose.pt）
- バッチサイズの削減
- 画像のリサイズ

## トラブルシューティング

### よくある問題

#### 1. カメラが開かない
```bash
# カメラIDを変更
python webcam_inference.py --camera 1
```

#### 2. モデルの読み込みエラー
```bash
# 事前学習済みモデルを使用
python webcam_inference.py --model yolo11n-pose.pt
```

#### 3. キーポイントが検出されない
- 手が画面内に完全に収まっているか確認
- 照明条件を改善
- 信頼度閾値を下げる（`--conf 0.3`）

#### 4. 処理が遅い
- GPU使用の確認
- 入力画像サイズの削減
- バッチサイズの調整

### デバッグモード
```bash
# 詳細なログ出力
python webcam_inference.py --verbose
```

## カスタマイズ

### 新しいジェスチャーの追加
1. `analyze_gesture` メソッドを編集
2. キーポイントの位置関係を定義
3. 閾値の調整

### モデルの微調整
1. `configs/train.yaml` の編集
2. 学習率、エポック数の調整
3. データ拡張パラメータの変更

### キーポイントの追加/変更
1. `keypoint_names` 辞書の編集
2. `skeleton` リストの更新
3. データセットの再作成

## 応用例

### 1. 手話翻訳システム
- リアルタイムでの手話認識
- 音声合成との連携
- 多言語対応

### 2. AR/VRアプリケーション
- ジェスチャーによる操作
- 3D空間での手の追跡
- 没入型インターフェース

### 3. リハビリテーション支援
- 手の動きの定量化
- 運動機能の評価
- トレーニング効果の測定

### 4. セキュリティシステム
- ジェスチャーベースの認証
- 異常行動の検出
- アクセス制御

## 参考資料

- [Ultralytics YOLO11で手のキーポイント推定を強化](https://www.ultralytics.com/ja/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11)
- [Ultralytics YOLO 公式ドキュメント](https://docs.ultralytics.com/)
- [YOLO11 ポーズ推定ガイド](https://docs.ultralytics.com/models/yolov11/)

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 貢献

バグレポート、機能要求、プルリクエストを歓迎します。問題や質問がある場合は、GitHubのIssuesページをご利用ください。
