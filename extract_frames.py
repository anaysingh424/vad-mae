import os
import cv2  # type: ignore
import glob
from tqdm import tqdm  # type: ignore

def extract_videos(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    videos = glob.glob(os.path.join(in_dir, "*.avi"))
    for vid_path in tqdm(videos):
        vid_name = os.path.splitext(os.path.basename(vid_path))[0]
        vid_out_dir = os.path.join(out_dir, vid_name)
        os.makedirs(vid_out_dir, exist_ok=True)
        cap = cv2.VideoCapture(vid_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(vid_out_dir, f"{frame_idx:04d}.png"), frame)
            frame_idx += 1
        cap.release()

if __name__ == '__main__':
    in_base_path = r"C:\Users\Anay\.gemini\antigravity\scratch\vad\Avenue_Extracted\Avenue Dataset"
    out_base_path = r"C:\Users\Anay\.gemini\antigravity\scratch\vad\Avenue_Extracted\Avenue Dataset"
    print("Extracting train videos...")
    extract_videos(os.path.join(in_base_path, "training_videos"), os.path.join(out_base_path, "train", "frames"))
    print("Extracting test videos...")
    extract_videos(os.path.join(in_base_path, "testing_videos"), os.path.join(out_base_path, "test", "frames"))
    print("Done")
