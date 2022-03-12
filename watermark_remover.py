import os
import sys

import cv2
import numpy
from moviepy import editor

class WatermarkRemover():
  
  def __init__(self, threshold: int, kernel_size: int):
    self.threshold = threshold #阈值分割所用阈值
    self.kernel_size = kernel_size #膨胀运算核尺寸
  
  def select_roi(self, img: numpy.ndarray, hint: str) -> list:
    '''
    框选水印或字幕位置，SPACE或ENTER键退出
    :param img: 显示图片
    :return: 框选区域坐标
    '''
    COFF = 0.5
    w, h = int(COFF * img.shape[1]), int(COFF * img.shape[0])
    resize_img = cv2.resize(img, (w, h))
    roi = cv2.selectROI(hint, resize_img, False, False)
    cv2.destroyAllWindows()
    watermark_roi = [int(roi[0] / COFF), int(roi[1] / COFF), int(roi[2] / COFF), int(roi[3] / COFF)]
    return watermark_roi

  def dilate_mask(self, mask: numpy.ndarray) -> numpy.ndarray:
    
    '''
    对蒙版进行膨胀运算
    :param mask: 蒙版图片
    :return: 膨胀处理后蒙版
    '''
    kernel = numpy.ones((self.kernel_size, self.kernel_size), numpy.uint8)
    mask = cv2.dilate(mask, kernel)
    return mask

  def generate_single_mask(self, img: numpy.ndarray, roi: list, threshold: int) -> numpy.ndarray:
    '''
    通过手动选择的ROI区域生成单帧图像的水印蒙版
    :param img: 单帧图像
    :param roi: 手动选择区域坐标
    :param threshold: 二值化阈值
    :return: 水印蒙版
    '''
    #区域无效，程序退出
    if len(roi) != 4:
      print('NULL ROI!')
      sys.exit()
    
    #复制单帧灰度图像ROI内像素点
    roi_img = numpy.zeros((img.shape[0], img.shape[1]), numpy.uint8)
    start_x, end_x = int(roi[1]), int(roi[1] + roi[3])
    start_y, end_y = int(roi[0]), int(roi[0] + roi[2])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_img[start_x:end_x, start_y:end_y] = gray[start_x:end_x, start_y:end_y]

    #阈值分割
    _, mask = cv2.threshold(roi_img, threshold, 255, cv2.THRESH_BINARY)
    return mask

  def generate_watermark_mask(self, video_path: str) -> numpy.ndarray:
    '''
    截取视频中多帧图像生成多张水印蒙版，通过逻辑与计算生成最终水印蒙版
    :param video_path: 视频文件路径
    :return: 水印蒙版
    '''
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    roi = self.select_roi(frame, 'select watermark ROI')
    mask = numpy.ones((frame.shape[0], frame.shape[1]), numpy.uint8)
    mask.fill(255)

    step = video.get(cv2.CAP_PROP_FRAME_COUNT) // 5
    index = 0
    while success:
      if index % step == 0:
        mask = cv2.bitwise_and(mask, self.generate_single_mask(frame, roi, self.threshold))
      success, frame = video.read()
      index += 1
    video.release()

    return self.dilate_mask(mask)

  def generate_subtitle_mask(self, frame: numpy.ndarray, roi: list) -> numpy.ndarray:
    '''
    通过手动选择ROI区域生成单帧图像字幕蒙版
    :param frame: 单帧图像
    :param roi: 手动选择区域坐标
    :return: 字幕蒙版
    '''
    mask = self.generate_single_mask(frame, [0, roi[1], frame.shape[1], roi[3]], self.threshold) #仅使用ROI横坐标区域
    return self.dilate_mask(mask)

  def inpaint_image(self, img: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    '''
    修复图像
    :param img: 单帧图像
    :parma mask: 蒙版
    :return: 修复后图像
    '''
    telea = cv2.inpaint(img, mask, 1, cv2.INPAINT_TELEA)
    return telea
  
  def merge_audio(self, input_path: str, output_path: str, temp_path: str):
    '''
    合并音频与处理后视频
    :param input_path: 原视频文件路径
    :param output_path: 封装音视频后文件路径
    :param temp_path: 无声视频文件路径 
    '''
    with editor.VideoFileClip(input_path) as video:
      audio = video.audio
      with editor.VideoFileClip(temp_path) as opencv_video:
        clip = opencv_video.set_audio(audio)
        clip.to_videofile(output_path)

  def remove_video_watermark(self, input_path: str):
    '''
    去除视频水印
    :param input_path: 视频文件路径
    '''
    #生成水印蒙版
    mask = self.generate_watermark_mask(input_path)

    #创建待写入文件对象
    video = cv2.VideoCapture(input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    temp_path = 'temp.mp4'
    video_writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    #逐帧处理图像
    success, frame = video.read()

    while success:
      frame = self.inpaint_image(frame, mask)
      video_writer.write(frame)
      success, frame = video.read()

    video.release()
    video_writer.release()

    #封装视频
    (_, filename) = os.path.split(input_path)
    output_path = filename.split('.')[0] + '_no_watermark.mp4'#输出文件路径
    self.merge_audio(input_path, output_path, temp_path)
    os.remove(temp_path)

  def remove_video_subtitle(self, input_path: str):
    '''
    去除视频字幕
    :param input_path: 视频文件路径
    '''
    #创建待写入文件对象
    video = cv2.VideoCapture(input_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    temp_path = 'temp.mp4'
    video_writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    
    #逐帧处理图像
    success, frame = video.read()
    roi = self.select_roi(frame, 'select subtitle ROI')

    while success:
      mask = self.generate_subtitle_mask(frame, roi)
      frame = self.inpaint_image(frame, mask)
      video_writer.write(frame)
      success, frame = video.read()

    video.release()
    video_writer.release()

    #封装视频
    (_, filename) = os.path.split(input_path)
    output_path = filename.split('.')[0] + '_no_sub.mp4'#输出文件路径
    self.merge_audio(input_path, output_path, temp_path)
    os.remove(temp_path)

if __name__ == '__main__':
  video_path = './video.mp4'#视频文件路径

  #去除视频水印
  remover = WatermarkRemover(threshold=80, kernel_size=5)
  remover.remove_video_watermark(video_path)

  #去除视频字幕
  remover = WatermarkRemover(threshold=80, kernel_size=10)
  remover.remove_video_subtitle(video_path)