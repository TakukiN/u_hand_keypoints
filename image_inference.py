#!/usr/bin/env python3
"""
画像での手のキーポイント推定
学習済みのYOLO11モデルを使用して画像から手のキーポイントを検出します
"""

import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path
import json

class ImageHandKeypointDetector:
    """画像での手のキーポイント検出器クラス"""
    
    def __init__(self, model_path='best.pt', device='auto'):
        """
        初期化
        
        Args:
            model_path (str): 学習済みモデルのパス
            device (str): 使用デバイス（'cpu', 'cuda', 'auto'）
        """
        self.model_path = model_path
        self.device = device
        self.model = None
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
        
        # キーポイントの接続（骨格表示用）
        self.skeleton = [
            # 親指
            (0, 3), (3, 2), (2, 1),
            # 人差指
            (0, 7), (7, 6), (6, 5), (5, 4),
            # 中指
            (0, 11), (11, 10), (10, 9), (9, 8),
            # 薬指
            (0, 15), (15, 14), (14, 13), (13, 12),
            # 小指
            (0, 19), (19, 18), (18, 17), (17, 16),
            # 手のひら
            (7, 11), (11, 15), (15, 19)
        ]
        
        self.load_model()
    
    def load_model(self):
        """モデルを読み込み"""
        try:
            print(f"モデルを読み込み中: {self.model_path}")
            
            # モデルファイルの存在確認
            if not Path(self.model_path).exists():
                print(f"警告: モデルファイル {self.model_path} が見つかりません。")
                print("事前学習済みのYOLO11ポーズ推定モデルを使用します。")
                self.model = YOLO('yolo11n-pose.pt')
            else:
                self.model = YOLO(self.model_path)
            
            print("モデルの読み込みが完了しました。")
            
        except Exception as e:
            print(f"モデルの読み込みに失敗しました: {e}")
            print("事前学習済みのYOLO11ポーズ推定モデルを使用します。")
            self.model = YOLO('yolo11n-pose.pt')
    
    def detect_keypoints(self, image):
        """
        画像から手のキーポイントを検出
        
        Args:
            image: 入力画像
            
        Returns:
            list: 検出されたキーポイントのリスト
        """
        try:
            # モデルで推論実行
            results = self.model(image, verbose=False)
            
            # 結果からキーポイントを抽出
            keypoints = []
            for result in results:
                if result.keypoints is not None:
                    kpts = result.keypoints.data.cpu().numpy()
                    for kpt in kpts:
                        keypoints.append(kpt)
            
            return keypoints
            
        except Exception as e:
            print(f"キーポイント検出中にエラーが発生しました: {e}")
            return []
    
    def draw_keypoints(self, image, keypoints, conf_threshold=0.5):
        """
        画像にキーポイントと骨格を描画
        
        Args:
            image: 描画対象の画像
            keypoints: キーポイントのリスト
            conf_threshold: 信頼度閾値
        """
        if not keypoints:
            return image
        
        # 画像のコピーを作成
        result_image = image.copy()
        
        # キーポイントの描画
        for kpts in keypoints:
            # 各キーポイントを描画
            for i, (x, y, conf) in enumerate(kpts):
                if conf > conf_threshold:
                    # キーポイントの位置を描画
                    cv2.circle(result_image, (int(x), int(y)), 8, (0, 255, 0), -1)
                    cv2.circle(result_image, (int(x), int(y)), 8, (0, 0, 0), 2)
                    
                    # キーポイント名を表示
                    if i < len(self.keypoint_names):
                        name = self.keypoint_names[i]
                        cv2.putText(result_image, name, (int(x) + 10, int(y) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        cv2.putText(result_image, name, (int(x) + 10, int(y) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            
            # 骨格の描画
            for start_idx, end_idx in self.skeleton:
                if (start_idx < len(kpts) and end_idx < len(kpts) and
                    kpts[start_idx][2] > conf_threshold and kpts[end_idx][2] > conf_threshold):
                    
                    start_point = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                    end_point = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                    
                    cv2.line(result_image, start_point, end_point, (255, 0, 0), 3)
                    cv2.line(result_image, start_point, end_point, (0, 0, 0), 1)
        
        return result_image
    
    def analyze_gesture(self, keypoints, conf_threshold=0.5):
        """
        キーポイントからジェスチャーを分析
        
        Args:
            keypoints: キーポイントのリスト
            conf_threshold: 信頼度閾値
            
        Returns:
            dict: 検出されたジェスチャーと詳細情報
        """
        if not keypoints:
            return {"gesture": "No hand detected", "confidence": 0.0, "details": {}}
        
        gesture_info = {
            "gesture": "Unknown",
            "confidence": 0.0,
            "details": {}
        }
        
        for kpts in keypoints:
            # 信頼度の高いキーポイントのみを使用
            valid_kpts = [kpt for kpt in kpts if kpt[2] > conf_threshold]
            
            if len(valid_kpts) < 5:  # 最低5個のキーポイントが必要
                continue
            
            # 平均信頼度を計算
            avg_conf = np.mean([kpt[2] for kpt in valid_kpts])
            gesture_info["confidence"] = avg_conf
            
            # 簡単なジェスチャー認識
            # 親指先と人差指先の距離でOKサインを検出
            if (len(valid_kpts) > 4 and 
                valid_kpts[1][2] > conf_threshold and  # 親指先
                valid_kpts[4][2] > conf_threshold):    # 人差指先
                
                thumb_tip = np.array([valid_kpts[1][0], valid_kpts[1][1]])
                index_tip = np.array([valid_kpts[4][0], valid_kpts[4][1]])
                
                distance = np.linalg.norm(thumb_tip - index_tip)
                gesture_info["details"]["thumb_index_distance"] = float(distance)
                
                if distance < 30:  # 距離が近い場合
                    gesture_info["gesture"] = "OK Sign"
                elif distance > 100:  # 距離が遠い場合
                    gesture_info["gesture"] = "Open Hand"
            
            # 指の伸び具合で数値ジェスチャーを検出
            finger_tips = [1, 4, 8, 12, 16]  # 各指の先端
            extended_fingers = 0
            finger_states = {}
            
            for tip_idx in finger_tips:
                if tip_idx < len(valid_kpts) and valid_kpts[tip_idx][2] > conf_threshold:
                    # 手首との位置関係で指が伸びているか判断
                    wrist_y = valid_kpts[0][1] if valid_kpts[0][2] > conf_threshold else 0
                    tip_y = valid_kpts[tip_idx][1]
                    
                    is_extended = tip_y < wrist_y
                    finger_states[self.keypoint_names[tip_idx]] = is_extended
                    
                    if is_extended:  # 指先が手首より上にある場合
                        extended_fingers += 1
            
            gesture_info["details"]["finger_states"] = finger_states
            gesture_info["details"]["extended_fingers"] = extended_fingers
            
            if extended_fingers == 1:
                gesture_info["gesture"] = "Pointing (1)"
            elif extended_fingers == 2:
                gesture_info["gesture"] = "Peace Sign (2)"
            elif extended_fingers == 3:
                gesture_info["gesture"] = "Three (3)"
            elif extended_fingers == 4:
                gesture_info["gesture"] = "Four (4)"
            elif extended_fingers == 5:
                gesture_info["gesture"] = "Open Hand (5)"
        
        return gesture_info
    
    def process_image(self, image_path, output_path=None, conf_threshold=0.5, save_json=False):
        """
        画像を処理してキーポイントを検出
        
        Args:
            image_path (str): 入力画像のパス
            output_path (str): 出力画像のパス（Noneの場合は自動生成）
            conf_threshold (float): 信頼度閾値
            save_json (bool): JSONファイルとして結果を保存するか
            
        Returns:
            dict: 検出結果
        """
        # 画像を読み込み
        if not Path(image_path).exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
        
        print(f"画像を読み込みました: {image_path}")
        print(f"画像サイズ: {image.shape[1]}x{image.shape[0]}")
        
        # キーポイント検出
        keypoints = self.detect_keypoints(image)
        
        if not keypoints:
            print("手のキーポイントが検出されませんでした。")
            return {"error": "No keypoints detected"}
        
        print(f"{len(keypoints)}個の手のキーポイントセットを検出しました。")
        
        # ジェスチャー分析
        gesture_info = self.analyze_gesture(keypoints, conf_threshold)
        
        # 結果を表示
        print(f"検出されたジェスチャー: {gesture_info['gesture']}")
        print(f"信頼度: {gesture_info['confidence']:.3f}")
        
        if gesture_info['details']:
            print("詳細情報:")
            for key, value in gesture_info['details'].items():
                print(f"  {key}: {value}")
        
        # キーポイントと骨格を描画
        result_image = self.draw_keypoints(image, keypoints, conf_threshold)
        
        # 結果画像を保存
        if output_path is None:
            input_name = Path(image_path).stem
            output_path = f"output_{input_name}_keypoints.jpg"
        
        cv2.imwrite(output_path, result_image)
        print(f"結果画像を保存しました: {output_path}")
        
        # JSONファイルとして結果を保存
        if save_json:
            json_path = output_path.replace('.jpg', '.json')
            result_data = {
                "input_image": image_path,
                "output_image": output_path,
                "detection_time": str(Path(image_path).stat().st_mtime),
                "keypoints": keypoints.tolist() if hasattr(keypoints, 'tolist') else keypoints,
                "gesture_analysis": gesture_info,
                "confidence_threshold": conf_threshold
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            print(f"結果データをJSONファイルに保存しました: {json_path}")
        
        return {
            "keypoints": keypoints,
            "gesture_info": gesture_info,
            "output_image": output_path
        }
    
    def visualize_keypoints(self, image_path, keypoints, conf_threshold=0.5):
        """
        キーポイントを可視化（matplotlib使用）
        
        Args:
            image_path (str): 画像パス
            keypoints: キーポイントのリスト
            conf_threshold (float): 信頼度閾値
        """
        if not keypoints:
            print("可視化するキーポイントがありません。")
            return
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # プロット作成
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 元画像
        ax1.imshow(image_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # キーポイント表示
        ax2.imshow(image_rgb)
        ax2.set_title('Detected Keypoints')
        ax2.axis('off')
        
        # キーポイントを描画
        for kpts in keypoints:
            # 各キーポイントを描画
            for i, (x, y, conf) in enumerate(kpts):
                if conf > conf_threshold:
                    ax2.scatter(x, y, c='red', s=50, alpha=0.7)
                    
                    # キーポイント名を表示
                    if i < len(self.keypoint_names):
                        name = self.keypoint_names[i]
                        ax2.annotate(name, (x, y), xytext=(5, 5), 
                                    textcoords='offset points', fontsize=8,
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # 骨格を描画
            for start_idx, end_idx in self.skeleton:
                if (start_idx < len(kpts) and end_idx < len(kpts) and
                    kpts[start_idx][2] > conf_threshold and kpts[end_idx][2] > conf_threshold):
                    
                    start_point = (kpts[start_idx][0], kpts[start_idx][1])
                    end_point = (kpts[end_idx][0], kpts[end_idx][1])
                    
                    ax2.plot([start_point[0], end_point[0]], 
                            [start_point[1], end_point[1]], 'b-', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        
        # 保存
        output_path = f"visualization_{Path(image_path).stem}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"可視化結果を保存しました: {output_path}")
        
        plt.show()

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='画像での手のキーポイント推定')
    parser.add_argument('--image', type=str, required=True,
                       help='入力画像のパス')
    parser.add_argument('--model', type=str, default='best.pt',
                       help='学習済みモデルのパス (default: best.pt)')
    parser.add_argument('--output', type=str, default=None,
                       help='出力画像のパス (default: auto-generated)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='信頼度閾値 (default: 0.5)')
    parser.add_argument('--device', type=str, default='auto',
                       help='使用デバイス (default: auto)')
    parser.add_argument('--save-json', action='store_true',
                       help='結果をJSONファイルとして保存')
    parser.add_argument('--visualize', action='store_true',
                       help='matplotlibで可視化')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("画像手のキーポイント推定システム")
    print("=" * 60)
    
    try:
        # 検出器を初期化
        detector = ImageHandKeypointDetector(
            model_path=args.model,
            device=args.device
        )
        
        # 画像処理
        result = detector.process_image(
            image_path=args.image,
            output_path=args.output,
            conf_threshold=args.conf,
            save_json=args.save_json
        )
        
        # 可視化
        if args.visualize and 'keypoints' in result:
            detector.visualize_keypoints(args.image, result['keypoints'], args.conf)
        
        print("\n処理が完了しました！")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
