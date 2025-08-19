import os
import imageio
from PIL import Image
import numpy as np
import cv2

class FrameExtractor:
    def __init__(self, video_path, output_dir='frames/'):
        self.video_path = video_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"[Error] Could not open video: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            print("[Warning] FPS not found, defaulting to 30.")
            fps = 30

        step = int(fps // 8)  # 2 fps

        frame_idx = 0
        saved_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.save(os.path.join(self.output_dir, f"frame_{frame_idx:04d}.png"))
                saved_idx += 1

            frame_idx += 1

        cap.release()
        print(f"[FrameExtractor] Extracted {saved_idx} frames at 2 fps.")

    @staticmethod
    def load_grayscale_frame(path):
        img = Image.open(path).convert('L')
        return np.array(img)