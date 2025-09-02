#!/usr/bin/env python3
"""
手のキーポイント推定用データセット作成ユーティリティ
画像とラベルファイルからYOLO形式のデータセットを作成します
"""

import os
import json
import yaml
from pathlib import Path
import argparse
import shutil
from typing import List, Dict, Tuple
import random

class HandKeypointDatasetCreator:
    """手のキーポイント推定用データセット作成クラス"""
    
    def __init__(self, dataset_path: str = "./data/hand_keypoints"):
        """
        初期化
        
        Args:
            dataset_path (str): データセットのルートパス
        """
        self.dataset_path = Path(dataset_path)
        self.keypoint_names = {
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
        
        # データセットディレクトリ構造を作成
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """データセットのディレクトリ構造を作成"""
        # メインディレクトリ
        (self.dataset_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
        (self.dataset_path / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
        
        print(f"データセットディレクトリ構造を作成しました: {self.dataset_path}")
    
    def create_dataset_config(self, output_path: str = None):
        """
        データセット設定ファイルを作成
        
        Args:
            output_path (str): 出力パス（Noneの場合はデフォルト）
            
        Returns:
            str: 作成された設定ファイルのパス
        """
        if output_path is None:
            output_path = self.dataset_path / "dataset.yaml"
        
        config = {
            'path': str(self.dataset_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': self.keypoint_names
        }
        
        # 設定ファイルを保存
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"データセット設定ファイルを作成しました: {output_path}")
        return str(output_path)
    
    def convert_keypoints_to_yolo_format(self, keypoints: List[List[float]], 
                                       img_width: int, img_height: int) -> str:
        """
        キーポイントをYOLO形式に変換
        
        Args:
            keypoints: キーポイントのリスト [[x, y, conf], ...]
            img_width: 画像の幅
            img_height: 画像の高さ
            
        Returns:
            str: YOLO形式のラベル文字列
        """
        if not keypoints or len(keypoints) < 21:
            return ""
        
        # 手のクラスID（0: 手）
        class_id = 0
        
        # バウンディングボックスを計算（手の領域）
        x_coords = [kp[0] for kp in keypoints if kp[2] > 0.5]
        y_coords = [kp[1] for kp in keypoints if kp[2] > 0.5]
        
        if not x_coords or not y_coords:
            return ""
        
        # バウンディングボックスの中心とサイズを計算
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # パディングを追加
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_width, x_max + padding)
        y_max = min(img_height, y_max + padding)
        
        # YOLO形式に変換（正規化）
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # ラベル行を作成
        label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        
        # キーポイントを追加
        for kp in keypoints:
            x_norm = kp[0] / img_width
            y_norm = kp[1] / img_height
            conf = kp[2]
            label_line += f" {x_norm:.6f} {y_norm:.6f} {conf:.6f}"
        
        return label_line
    
    def process_annotation_file(self, annotation_path: str, 
                               output_dir: str, 
                               split_ratio: Tuple[float, float, float] = (0.7, 0.2, 0.1)):
        """
        アノテーションファイルを処理してYOLO形式に変換
        
        Args:
            annotation_path (str): アノテーションファイルのパス（JSON形式）
            output_dir (str): 出力ディレクトリ
            split_ratio (tuple): 訓練/検証/テストの分割比率
        """
        if not Path(annotation_path).exists():
            print(f"エラー: アノテーションファイルが見つかりません: {annotation_path}")
            return
        
        # アノテーションファイルを読み込み
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"アノテーションファイルを読み込みました: {len(annotations)}件")
        
        # データを分割
        random.shuffle(annotations)
        total = len(annotations)
        train_end = int(total * split_ratio[0])
        val_end = train_end + int(total * split_ratio[1])
        
        train_data = annotations[:train_end]
        val_data = annotations[train_end:val_end]
        test_data = annotations[val_end:]
        
        print(f"データ分割: 訓練={len(train_data)}, 検証={len(val_data)}, テスト={len(test_data)}")
        
        # 各分割を処理
        self._process_split(train_data, "train", output_dir)
        self._process_split(val_data, "val", output_dir)
        self._process_split(test_data, "test", output_dir)
    
    def _process_split(self, data: List[Dict], split_name: str, output_dir: str):
        """
        データ分割を処理
        
        Args:
            data: 処理するデータ
            split_name: 分割名（train/val/test）
            output_dir: 出力ディレクトリ
        """
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for i, item in enumerate(data):
            try:
                # 画像ファイルのパスを取得
                image_path = item.get('image_path', item.get('file_name', ''))
                if not image_path or not Path(image_path).exists():
                    print(f"警告: 画像ファイルが見つかりません: {image_path}")
                    continue
                
                # キーポイントを取得
                keypoints = item.get('keypoints', item.get('landmarks', []))
                if not keypoints:
                    print(f"警告: キーポイントが見つかりません: {image_path}")
                    continue
                
                # 画像を読み込んでサイズを取得
                import cv2
                img = cv2.imread(image_path)
                if img is None:
                    print(f"警告: 画像の読み込みに失敗しました: {image_path}")
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # YOLO形式に変換
                yolo_label = self.convert_keypoints_to_yolo_format(keypoints, img_width, img_height)
                if not yolo_label:
                    print(f"警告: ラベルの変換に失敗しました: {image_path}")
                    continue
                
                # ファイル名を生成
                base_name = Path(image_path).stem
                new_image_name = f"{split_name}_{i:04d}_{base_name}.jpg"
                new_label_name = f"{split_name}_{i:04d}_{base_name}.txt"
                
                # 画像とラベルをコピー
                new_image_path = split_dir / "images" / new_image_name
                new_label_path = split_dir / "labels" / new_label_name
                
                new_image_path.parent.mkdir(parents=True, exist_ok=True)
                new_label_path.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(image_path, new_image_path)
                
                with open(new_label_path, 'w', encoding='utf-8') as f:
                    f.write(yolo_label)
                
                print(f"処理完了: {new_image_name}")
                
            except Exception as e:
                print(f"エラー: {image_path} の処理中にエラーが発生しました: {e}")
    
    def create_sample_dataset(self):
        """サンプルデータセットを作成（テスト用）"""
        print("サンプルデータセットを作成中...")
        
        # サンプル画像とラベルを作成
        sample_data = []
        
        # 簡単なサンプルデータを生成
        for i in range(10):
            sample_item = {
                'image_path': f'sample_image_{i}.jpg',
                'keypoints': [
                    [100 + i*10, 100 + i*5, 0.9],   # 手首
                    [120 + i*10, 80 + i*5, 0.8],    # 親指先
                    [110 + i*10, 90 + i*5, 0.8],    # 親指第2関節
                    [105 + i*10, 95 + i*5, 0.8],    # 親指第1関節
                    [130 + i*10, 70 + i*5, 0.8],    # 人差指先
                    [125 + i*10, 80 + i*5, 0.8],    # 人差指第2関節
                    [120 + i*10, 85 + i*5, 0.8],    # 人差指第1関節
                    [115 + i*10, 90 + i*5, 0.8],    # 人差指付け根
                    [140 + i*10, 75 + i*5, 0.8],    # 中指先
                    [135 + i*10, 85 + i*5, 0.8],    # 中指第2関節
                    [130 + i*10, 90 + i*5, 0.8],    # 中指第1関節
                    [125 + i*10, 95 + i*5, 0.8],    # 中指付け根
                    [150 + i*10, 80 + i*5, 0.8],    # 薬指先
                    [145 + i*10, 90 + i*5, 0.8],    # 薬指第2関節
                    [140 + i*10, 95 + i*5, 0.8],    # 薬指第1関節
                    [135 + i*10, 100 + i*5, 0.8],   # 薬指付け根
                    [160 + i*10, 85 + i*5, 0.8],    # 小指先
                    [155 + i*10, 95 + i*5, 0.8],    # 小指第2関節
                    [150 + i*10, 100 + i*5, 0.8],   # 小指第1関節
                    [145 + i*10, 105 + i*5, 0.8],   # 小指付け根
                    [110 + i*10, 100 + i*5, 0.8],   # 手のひら中心
                ]
            }
            sample_data.append(sample_item)
        
        # サンプルデータを処理
        self._process_split(sample_data, "train", str(self.dataset_path))
        self._process_split(sample_data, "val", str(self.dataset_path))
        self._process_split(sample_data, "test", str(self.dataset_path))
        
        print("サンプルデータセットの作成が完了しました。")
    
    def validate_dataset(self):
        """データセットの検証を実行"""
        print("データセットの検証を実行中...")
        
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not images_dir.exists() or not labels_dir.exists():
                print(f"警告: {split} ディレクトリが存在しません")
                continue
            
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            label_files = list(labels_dir.glob('*.txt'))
            
            print(f"{split}: 画像={len(image_files)}, ラベル={len(label_files)}")
            
            # ラベルファイルの内容を検証
            for label_file in label_files[:5]:  # 最初の5件のみ検証
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            parts = content.split()
                            if len(parts) >= 5:  # 最低限の要素数
                                print(f"  {label_file.name}: OK")
                            else:
                                print(f"  {label_file.name}: 不正な形式")
                        else:
                            print(f"  {label_file.name}: 空ファイル")
                except Exception as e:
                    print(f"  {label_file.name}: エラー - {e}")
        
        print("データセット検証が完了しました。")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='手のキーポイント推定用データセット作成')
    parser.add_argument('--dataset-path', type=str, default='./data/hand_keypoints',
                       help='データセットのルートパス')
    parser.add_argument('--annotation-file', type=str, default=None,
                       help='アノテーションファイルのパス（JSON形式）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='出力ディレクトリ')
    parser.add_argument('--create-sample', action='store_true',
                       help='サンプルデータセットを作成')
    parser.add_argument('--validate', action='store_true',
                       help='データセットの検証を実行')
    parser.add_argument('--split-ratio', type=float, nargs=3, 
                       default=[0.7, 0.2, 0.1],
                       help='訓練/検証/テストの分割比率')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("手のキーポイント推定用データセット作成システム")
    print("=" * 60)
    
    # データセット作成器を初期化
    creator = HandKeypointDatasetCreator(args.dataset_path)
    
    # データセット設定ファイルを作成
    config_path = creator.create_dataset_config()
    print(f"設定ファイル: {config_path}")
    
    if args.create_sample:
        # サンプルデータセットを作成
        creator.create_sample_dataset()
    
    if args.annotation_file:
        # アノテーションファイルを処理
        output_dir = args.output_dir or args.dataset_path
        creator.process_annotation_file(
            args.annotation_file, 
            output_dir, 
            tuple(args.split_ratio)
        )
    
    if args.validate:
        # データセットを検証
        creator.validate_dataset()
    
    print("\nデータセット作成が完了しました！")
    print(f"データセットパス: {args.dataset_path}")
    print(f"設定ファイル: {config_path}")

if __name__ == "__main__":
    main()
