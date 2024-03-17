import cv2
import numpy as np
import glob
from moviepy.editor import VideoFileClip
import os
from tqdm import tqdm
import argparse

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

def process_video(video_clip, output_path, apply_mask_func, file_ext='mp4', audio_ext=None, threads=None, audio_codec=None, crf=23):
    # Convert threads from integer to string unless None
    if(threads != None):
        threads = str(threads)
    # Create list of FFMPEG parameters
    ffmpeg_params = ['-crf',str(crf)]
    # If there is a specified file extension for audio, add it to parameter list
    if(audio_ext != None):
        ffmpeg_params.append('-c:a')
        ffmpeg_params.append(audio_ext)

    total_frames = int(video_clip.duration * video_clip.fps)
    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frames")

    def process_frame(frame):
        result = apply_mask_func(frame)
        progress_bar.update(1000)
        return result
    
    processed_video = video_clip.fl_image(process_frame, apply_to=["each"])
    processed_video.write_videofile(f"{output_path}.{file_ext}", codec="libx264", threads=threads, audio_codec=audio_codec, ffmpeg_params=ffmpeg_params)

if __name__ == "__main__":
    # Parse input arguments
    parser = argparse.ArgumentParser("watermark_remover.py")
    # TODO: Add all file types that support x264 codec to choices
    parser.add_argument('-e', '--file_ext', required=False, choices=['mp4','mkv'], help='File extension to save output video with. Default: mp4', default='mp4')
    parser.add_argument('-c', '--crf', required=False, type=int, choices=range(0,51+1), metavar="[0,51]", help='Quality of output video. 0 is lossless, whereas 51 has the lowest bitrate. Default: 23', default=23)
    # TODO: Added range so number of threads cannot be <1, but 99 is an arbitrary upper limit
    parser.add_argument('-t', '--threads', required=False, type=int, choices=range(1,100), metavar="[1,99]", help='Number of threads to process output video with. Should be 2x the number of your CPU cores. Default: None (Allows MoviePy to decide)', default=None)
    parser.add_argument('-ac', '--audio_codec', required=False, choices=['libmp3lame','libvorbis','libfdk_aac','pcm_s16le','pcm_s32le'], help='Codec for audio in output video. Default: None (Allows MoviePy to decide)', default=None)
    # TODO: Add all audio file types supported by above codecs
    parser.add_argument('-ae', '--audio_ext', required=False, choices=['mp3','aac','wav','flac'], help='File extension for audio in output video. Choose a format supported by your selected codec. Default: None (Allows FFMPEG to decide)', default=None)
    args = vars(parser.parse_args())

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
        process_video(video_clip, output_video_path, mask_func, threads=args['threads'], audio_ext=args['audio_ext'], audio_codec=args['audio_codec'], file_ext=args['file_ext'], crf=args['crf'])
        print(f"Successfully processed {video_name}")
