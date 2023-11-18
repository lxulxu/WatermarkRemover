import cv2
import numpy as np
import glob
from moviepy.editor import VideoFileClip
import os
from tqdm import tqdm

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        except OSError as error:
            print(f"Error creating directory {directory}: {error}")
            raise

def is_valid_video_file(file):
    try:
        with VideoFileClip(file) as video_clip:
            return True
    except Exception as e:
        print(f"Invalid video file: {file}, Error: {e}")
        return False

def get_first_valid_frame(video_clip, threshold=10, num_frames=10):
    total_frames = int(video_clip.fps * video_clip.duration)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    for idx in frame_indices:
        frame = video_clip.get_frame(idx / video_clip.fps)
        if frame.mean() > threshold:
            return frame

    return video_clip.get_frame(0)

def select_roi_for_mask(video_clip):
    frame = get_first_valid_frame(video_clip)

    # 将视频帧调整为720p显示
    display_height = 720
    scale_factor = display_height / frame.shape[0]
    display_width = int(frame.shape[1] * scale_factor)
    display_frame = cv2.resize(frame, (display_width, display_height))

    instructions = "Select ROI and press SPACE or ENTER"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display_frame, instructions, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    r = cv2.selectROI(display_frame)
    cv2.destroyAllWindows()

    r_original = (int(r[0] / scale_factor), int(r[1] / scale_factor), int(r[2] / scale_factor), int(r[3] / scale_factor))

    return r_original

def detect_watermark_adaptive(frame, roi):
    roi_frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
    mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = binary_frame

    return mask

def generate_watermark_mask(video_clip, num_frames=10, min_frame_count=7):
    total_frames = int(video_clip.duration * video_clip.fps)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = [video_clip.get_frame(idx / video_clip.fps) for idx in frame_indices]
    r_original = select_roi_for_mask(video_clip)

    masks = [detect_watermark_adaptive(frame, r_original) for frame in frames]

    final_mask = sum((mask == 255).astype(np.uint8) for mask in masks)
    # 根据像素点在至少min_frame_count张以上的帧中的出现来生成最终的遮罩
    final_mask = np.where(final_mask >= min_frame_count, 255, 0).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(final_mask, kernel)

def process_video(video_clip, output_path, apply_mask_func):
    total_frames = int(video_clip.duration * video_clip.fps)
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frames")

    def process_frame(frame):
        result = apply_mask_func(frame)
        progress_bar.update(1000)
        return result
    
    processed_video = video_clip.fl_image(process_frame, apply_to=["each"])
    processed_video.write_videofile(f"{output_path}.mp4", codec="libx264")

if __name__ == "__main__":
    output_dir = "output"
    ensure_directory_exists(output_dir)
    videos = [f for f in glob.glob("video/*") if is_valid_video_file(f)]

    watermark_mask = None

    for video in videos:
        video_clip = VideoFileClip(video)
        if watermark_mask is None:
            watermark_mask = generate_watermark_mask(video_clip)

        mask_func = lambda frame: cv2.inpaint(frame, watermark_mask, 3, cv2.INPAINT_NS)
        video_name = os.path.basename(video)
        output_video_path = os.path.join(output_dir, os.path.splitext(video_name)[0])
        process_video(video_clip, output_video_path, mask_func)
        print(f"Successfully processed {video_name}")