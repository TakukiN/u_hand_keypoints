#!/usr/bin/env python3
"""
ウェブカメラでのリアルタイム手のキーポイント推定
学習済みのYOLO11モデルを使用してウェブカメラから手のキーポイントを検出します
"""

import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import time
from pathlib import Path

class HandKeypointDetector:
    """手のキーポイント検出器クラス"""
    
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
    
    def detect_keypoints(self, frame):
        """
        フレームから手のキーポイントを検出
        
        Args:
            frame: 入力フレーム
            
        Returns:
            list: 検出されたキーポイントのリスト
        """
        try:
            # モデルで推論実行
            results = self.model(frame, verbose=False)
            
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
    
    def draw_keypoints(self, frame, keypoints, conf_threshold=0.5):
        """
        フレームにキーポイントと骨格を描画
        
        Args:
            frame: 描画対象のフレーム
            keypoints: キーポイントのリスト
            conf_threshold: 信頼度閾値
        """
        if not keypoints:
            return frame
        
        # キーポイントの描画
        for kpts in keypoints:
            # 各キーポイントを描画
            for i, (x, y, conf) in enumerate(kpts):
                if conf > conf_threshold:
                    # キーポイントの位置を描画
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                    
                    # キーポイント名を表示
                    if i < len(self.keypoint_names):
                        name = self.keypoint_names[i]
                        cv2.putText(frame, name, (int(x) + 10, int(y) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # 骨格の描画
            for start_idx, end_idx in self.skeleton:
                if (start_idx < len(kpts) and end_idx < len(kpts) and
                    kpts[start_idx][2] > conf_threshold and kpts[end_idx][2] > conf_threshold):
                    
                    start_point = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                    end_point = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                    
                    cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        
        return frame
    
    def analyze_gesture(self, keypoints, conf_threshold=0.5):
        """
        キーポイントからジェスチャーを分析
        
        Args:
            keypoints: キーポイントのリスト
            conf_threshold: 信頼度閾値
            
        Returns:
            str: 検出されたジェスチャー
        """
        if not keypoints:
            return "No hand detected"
        
        for kpts in keypoints:
            # 信頼度の高いキーポイントのみを使用
            valid_kpts = [kpt for kpt in kpts if kpt[2] > conf_threshold]
            
            if len(valid_kpts) < 5:  # 最低5個のキーポイントが必要
                continue
            
            # 簡単なジェスチャー認識
            # 親指先と人差指先の距離でOKサインを検出
            if (len(valid_kpts) > 4 and 
                valid_kpts[1][2] > conf_threshold and  # 親指先
                valid_kpts[4][2] > conf_threshold):    # 人差指先
                
                thumb_tip = np.array([valid_kpts[1][0], valid_kpts[1][1]])
                index_tip = np.array([valid_kpts[4][0], valid_kpts[4][1]])
                
                distance = np.linalg.norm(thumb_tip - index_tip)
                
                if distance < 30:  # 距離が近い場合
                    return "OK Sign"
                elif distance > 100:  # 距離が遠い場合
                    return "Open Hand"
            
            # 指の伸び具合で数値ジェスチャーを検出
            finger_tips = [1, 4, 8, 12, 16]  # 各指の先端
            extended_fingers = 0
            
            for tip_idx in finger_tips:
                if tip_idx < len(valid_kpts) and valid_kpts[tip_idx][2] > conf_threshold:
                    # 手首との位置関係で指が伸びているか判断
                    wrist_y = valid_kpts[0][1] if valid_kpts[0][2] > conf_threshold else 0
                    tip_y = valid_kpts[tip_idx][1]
                    
                    if tip_y < wrist_y:  # 指先が手首より上にある場合
                        extended_fingers += 1
            
            if extended_fingers == 1:
                return "Pointing (1)"
            elif extended_fingers == 2:
                return "Peace Sign (2)"
            elif extended_fingers == 3:
                return "Three (3)"
            elif extended_fingers == 4:
                return "Four (4)"
            elif extended_fingers == 5:
                return "Open Hand (5)"
        
        return "Unknown Gesture"
    
    def run_webcam(self, camera_id=0, conf_threshold=0.5):
        """
        ウェブカメラでリアルタイム推論を実行
        
        Args:
            camera_id (int): カメラID
            conf_threshold (float): 信頼度閾値
        """
        print(f"カメラ {camera_id} を起動中...")
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"エラー: カメラ {camera_id} を開けませんでした。")
            return
        
        print("ウェブカメラが起動しました。")
        print("操作:")
        print("  - 'q': 終了")
        print("  - 's': スクリーンショット保存")
        print("  - 'c': 信頼度閾値変更")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("フレームの読み込みに失敗しました。")
                    break
                
                # フレームをリサイズ（処理速度向上のため）
                frame = cv2.resize(frame, (640, 480))
                
                # キーポイント検出
                keypoints = self.detect_keypoints(frame)
                
                # キーポイントと骨格を描画
                frame = self.draw_keypoints(frame, keypoints, conf_threshold)
                
                # ジェスチャー分析
                gesture = self.analyze_gesture(keypoints, conf_threshold)
                
                # 情報表示
                cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Conf Threshold: {conf_threshold:.2f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # FPS計算
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    fps = 30 / elapsed_time
                    start_time = time.time()
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # フレーム表示
                cv2.imshow('Hand Keypoint Detection', frame)
                
                # キー入力処理
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # スクリーンショット保存
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"hand_keypoints_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"スクリーンショットを保存しました: {filename}")
                elif key == ord('c'):
                    # 信頼度閾値変更
                    new_threshold = float(input("新しい信頼度閾値を入力 (0.0-1.0): "))
                    if 0.0 <= new_threshold <= 1.0:
                        conf_threshold = new_threshold
                        print(f"信頼度閾値を {conf_threshold:.2f} に変更しました。")
                    else:
                        print("無効な値です。0.0から1.0の間で入力してください。")
        
        except KeyboardInterrupt:
            print("\nユーザーによって中断されました。")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("ウェブカメラを終了しました。")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='ウェブカメラでの手のキーポイント推定')
    parser.add_argument('--model', type=str, default='best.pt',
                       help='学習済みモデルのパス (default: best.pt)')
    parser.add_argument('--camera', type=int, default=0,
                       help='カメラID (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='信頼度閾値 (default: 0.5)')
    parser.add_argument('--device', type=str, default='auto',
                       help='使用デバイス (default: auto)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ウェブカメラ手のキーポイント推定システム")
    print("=" * 60)
    
    # 検出器を初期化
    detector = HandKeypointDetector(
        model_path=args.model,
        device=args.device
    )
    
    # ウェブカメラで実行
    detector.run_webcam(
        camera_id=args.camera,
        conf_threshold=args.conf
    )

if __name__ == "__main__":
    main()
