# WatermarkRemover
通过手动框选区域批量去除多个视频中位置固定的某个水印，项目基于Python 3.12。

## 效果
- 原始帧
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/origin.jpg'></a>
- 去除水印
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/no_watermark.jpg'></a>


## 如何使用

### 1. 安装依赖：
  `pip install -r requirements.txt`

### 2. 准备视频文件
  待处理视频放在`video`文件夹下，所有视频尺寸须保持一致。

### 3. 运行程序
  `python watermark_remover.py`
### 4.选择水印区域
  鼠标框选水印对应区域后按**SPACE**或**ENTER**键，处理后视频在`output`文件夹下，格式为mp4。