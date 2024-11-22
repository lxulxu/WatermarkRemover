# WatermarkRemover
通过手动框选区域批量去除多个视频中位置固定的某个水印，项目基于Python 3.12。

## 效果
<iframe frameborder="0" class="juxtapose" width="100%" height="540" src="https://cdn.knightlab.com/libs/juxtapose/latest/embed/index.html?uid=3db06fcc-a8aa-11ef-9397-d93975fe8866"></iframe>

## 如何使用

### 1. 安装依赖：
  `pip install -r requirements.txt`

### 2. 准备视频文件
  待处理视频放在`video`文件夹下，所有视频尺寸须保持一致。

### 3. 运行程序
  `python watermark_remover.py`
### 4.选择水印区域
  鼠标框选水印对应区域后按**SPACE**或**ENTER**键，处理后视频在`output`文件夹下，格式为mp4。