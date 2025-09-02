#!/usr/bin/env python3
"""
手のキーポイント推定モデルの学習スクリプト
Ultralytics YOLO11を使用して手のキーポイント推定モデルを学習します
"""

import os
import yaml
from ultralytics import YOLO
from pathlib import Path

def create_dataset_config():
    """データセット設定ファイルを作成"""
    config = {
        'path': './data/hand_keypoints',  # データセットのパス
        'train': 'images/train',          # 学習画像
        'val': 'images/val',              # 検証画像
        'test': 'images/test',            # テスト画像
        
        # 手のキーポイントクラス（21個のキーポイント）
        'names': {
            0: 'wrist',           # 手首
            1: 'thumb_tip',       # 親指先
            2: 'thumb_ip',        # 親指第2関節
            3: 'thumb_mcp',       # 親指第1関節
            4: 'index_tip',       # 人差指先
            5: 'index_dip',       # 人差指第2関節
            6: 'index_pip',       # 人差指第1関節
            7: 'index_mcp',       # 人差指付け根
            8: 'middle_tip',      # 中指先
            9: 'middle_dip',      # 中指第2関節
            10: 'middle_pip',     # 中指第1関節
            11: 'middle_mcp',     # 中指付け根
            12: 'ring_tip',       # 薬指先
            13: 'ring_dip',       # 薬指第2関節
            14: 'ring_pip',       # 薬指第1関節
            15: 'ring_mcp',       # 薬指付け根
            16: 'pinky_tip',      # 小指先
            17: 'pinky_dip',      # 小指第2関節
            18: 'pinky_pip',      # 小指第1関節
            19: 'pinky_mcp',      # 小指付け根
            20: 'palm_center'     # 手のひら中心
        }
    }
    
    # 設定ファイルを保存
    os.makedirs('./configs', exist_ok=True)
    with open('./configs/dataset.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return './configs/dataset.yaml'

def create_training_config():
    """学習設定ファイルを作成"""
    config = {
        'task': 'pose',           # ポーズ推定タスク
        'model': 'yolo11n-pose.pt',  # ベースモデル
        'data': './configs/dataset.yaml',
        'epochs': 100,            # エポック数
        'patience': 20,           # 早期停止の忍耐
        'batch': 16,              # バッチサイズ
        'imgsz': 640,             # 入力画像サイズ
        'device': 'auto',         # デバイス（GPU/CPU）
        'workers': 8,             # データローダーのワーカー数
        
        # 学習率設定
        'lr0': 0.01,             # 初期学習率
        'lrf': 0.01,             # 最終学習率
        'momentum': 0.937,        # モーメンタム
        'weight_decay': 0.0005,   # 重み減衰
        
        # データ拡張
        'hsv_h': 0.015,          # HSV色相
        'hsv_s': 0.7,            # HSV彩度
        'hsv_v': 0.4,            # HSV明度
        'degrees': 0.0,           # 回転
        'translate': 0.1,         # 平行移動
        'scale': 0.5,             # スケール
        'shear': 0.0,             # せん断
        'perspective': 0.0,       # 透視変換
        'flipud': 0.0,            # 上下反転
        'fliplr': 0.5,            # 左右反転
        'mosaic': 1.0,            # モザイク
        'mixup': 0.0,             # ミックスアップ
        
        # ポーズ推定特有の設定
        'conf': 0.001,            # 信頼度閾値
        'iou': 0.6,               # IoU閾値
        'max_det': 300,           # 最大検出数
        
        # 保存設定
        'save': True,             # モデル保存
        'save_period': 10,        # 保存間隔
        'project': './runs',      # プロジェクトディレクトリ
        'name': 'hand_keypoints', # 実験名
        'exist_ok': True,         # 既存ディレクトリ上書き
        'pretrained': True,       # 事前学習済み重み使用
        'optimizer': 'auto',      # 最適化アルゴリズム
        'verbose': True,          # 詳細出力
        'seed': 0,                # 乱数シード
        'deterministic': True,    # 決定論的動作
        'single_cls': False,      # 単一クラス
        'rect': False,            # 矩形学習
        'cos_lr': False,          # コサイン学習率スケジューリング
        'close_mosaic': 10,       # モザイク終了エポック
        'amp': True,              # 混合精度学習
        'fraction': 1.0,          # データセット使用率
        'cache': False,           # キャッシュ
        'overlap_mask': True,     # マスク重複
        'mask_ratio': 4,          # マスク比率
        'dropout': 0.0,           # ドロップアウト
        'val': True,              # 検証実行
        'plots': True,            # プロット生成
    }
    
    # 設定ファイルを保存
    with open('./configs/train.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    return './configs/train.yaml'

def prepare_dataset_structure():
    """データセットのディレクトリ構造を作成"""
    dataset_path = Path('./data/hand_keypoints')
    
    # ディレクトリ構造を作成
    (dataset_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (dataset_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (dataset_path / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (dataset_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (dataset_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    (dataset_path / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
    
    print(f"データセットディレクトリ構造を作成しました: {dataset_path}")
    print("以下のディレクトリにデータを配置してください:")
    print(f"  - 画像: {dataset_path}/images/")
    print(f"  - ラベル: {dataset_path}/labels/")

def train_model(config_file=None):
    """モデルの学習を実行"""
    print("手のキーポイント推定モデルの学習を開始します...")
    
    if config_file:
        # 指定された設定ファイルを読み込む
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # YOLOモデルを初期化
        model_path = config.get('model', 'yolo11n-pose.pt')
        model = YOLO(model_path)
        
        print(f"設定ファイル: {config_file}")
        print(f"データセット: {config.get('data')}")
        print(f"エポック数: {config.get('epochs')}")
        print(f"バッチサイズ: {config.get('batch')}")
        
        try:
            # モデルの学習を実行
            print("\n学習を開始します...")
            results = model.train(**config)
            
            print("学習が完了しました！")
            print(f"結果は {results.save_dir} に保存されました。")
            
        except Exception as e:
            print(f"学習中にエラーが発生しました: {e}")
            print("データセットの設定とデータの配置を確認してください。")
    else:
        # 設定ファイルを作成
        dataset_config = create_dataset_config()
        train_config = create_training_config()
        
        # データセット構造を準備
        prepare_dataset_structure()
        
        # YOLOモデルを初期化
        model = YOLO('yolo11n-pose.pt')
        
        print(f"データセット設定: {dataset_config}")
        print(f"学習設定: {train_config}")
        print("\n学習を開始する前に、データセットを適切に配置してください。")
        print("学習を開始しますか？ (y/n): ", end="")
        
        # ユーザー確認
        response = input().lower().strip()
        if response != 'y':
            print("学習をキャンセルしました。")
            return
        
        try:
            # モデルの学習を実行
            print("\n学習を開始します...")
            results = model.train(
                data=dataset_config,
                cfg=train_config,
                epochs=100,
                imgsz=640,
                batch=16,
                device='auto'
            )
            
            print("学習が完了しました！")
            print(f"結果は {results.save_dir} に保存されました。")
            
            # 学習結果の表示
            print("\n学習結果:")
            print(f"最終mAP: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"最終mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            
        except Exception as e:
            print(f"学習中にエラーが発生しました: {e}")
            print("データセットの設定とデータの配置を確認してください。")

def main():
    """メイン関数"""
    import argparse
    parser = argparse.ArgumentParser(description='手のキーポイント推定モデルの学習')
    parser.add_argument('--config', type=str, help='設定ファイルのパス')
    args = parser.parse_args()
    
    print("=" * 60)
    print("手のキーポイント推定モデル学習システム")
    print("=" * 60)
    
    # 必要なディレクトリを作成
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./configs', exist_ok=True)
    os.makedirs('./runs', exist_ok=True)
    
    # 学習を実行
    train_model(args.config)

if __name__ == "__main__":
    main()
