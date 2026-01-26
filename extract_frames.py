
import cv2
import numpy as np
import argparse
import os


def extract_and_merge_frames(video_path, output_path, num_frames=5, view_index=0, num_views=5):
    """
    動画ファイルから指定した数のフレームを等間隔で抽出します。
    動画が複数のカメラ視点を横に結合している場合、そのうちの1つだけを切り出します。
    
    Args:
        video_path (str): 入力動画ファイルのパス
        output_path (str): 出力画像ファイルのパス
        num_frames (int): 抽出するフレーム数 (時間の経過)
        view_index (int): 抽出するカメラ視点のインデックス (0始まり)
        num_views (int): 動画に含まれるカメラ視点の総数 (横並び)
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 1つの視点の幅を計算
    single_view_width = video_width // num_views
    print(f"Video Info: {video_width}x{video_height}, Total Frames: {total_frames}")
    print(f"Extracting View {view_index+1}/{num_views} (Width: {single_view_width})")

    # 抽出するフレームのインデックスを計算
    # 最初と最後は除外した方が安定する（初期化直後や終了直前を避けるため）
    margin = int(total_frames * 0.05)
    indices = np.linspace(margin, total_frames - margin, num_frames, dtype=int)
    
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Crop to the specific view
            # frame shape is (h, w, c)
            start_x = view_index * single_view_width
            end_x = start_x + single_view_width
            
            cropped_frame = frame[:, start_x:end_x, :]
            
            # 枠線を追加（オプション: 白い境界線）
            cropped_frame = cv2.copyMakeBorder(cropped_frame, 0, 0, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            
            frames.append(cropped_frame)
        else:
            print(f"Warning: Could not read frame at index {idx}")

    cap.release()

    if not frames:
        print("Error: No frames extracted")
        return

    # 横に結合 (Horizontal concatenation)
    merged_image = cv2.hconcat(frames)

    # 保存
    cv2.imwrite(output_path, merged_image)
    print(f"Saved merged image to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from multi-view video and merge them into one image.")
    parser.add_argument("video_path", type=str, help="Path to the input video file (mp4)")
    parser.add_argument("--output", type=str, default="frames_merged.png", help="Path to the output image file")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of time steps to extract")
    parser.add_argument("--view_index", type=int, default=0, help="Index of the camera view to extract (0-4 for 5 views)")
    parser.add_argument("--num_views", type=int, default=5, help="Total number of horizontally stacked views in the video")

    args = parser.parse_args()

    extract_and_merge_frames(args.video_path, args.output, args.num_frames, args.view_index, args.num_views)
