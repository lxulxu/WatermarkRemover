# WatermarkRemover
通过手动框选区域去除视频中位置固定的水印和字幕，项目基于Python3.7。

## 效果
- 原始帧
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/origin.jpg'></a>
- 去除水印
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/no_watermark.jpg'></a>
- 去除字幕
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/no_sub.jpg'></a>

## 如何使用

### 1. 安装依赖：
  `pip install -r requirements.txt`

### 2. 运行程序
待处理视频放在`video`文件夹下，所有视频尺寸须保持一致，鼠标框选水印或字幕对应区域后按**SPACE**或**ENTER**键

- **函数调用示例**
```python
#去除视频水印
remover = WatermarkRemover(threshold=80, kernel_size=5)
remover.remove_video_watermark(video_path)

#去除视频字幕
remover = WatermarkRemover(threshold=80, kernel_size=10)
remover.remove_video_subtitle(video_path)
```
| Param | Description |
| - | - |
| threshold | 阈值分割灰度值，范围0~255，根据水印灰度值自行调整 |
| kernel_size | 膨胀运算核尺寸，范围所有正整数，用于处理水印或字幕边缘 |

- **输出**
去除水印：`output/[文件名] + [_no_watermark].mp4`
去除字幕：`output/[文件名] + [_no_sub].mp4`

## 流程图
  - **去除水印**
    ```mermaid
    graph LR
    ROI[框选水印] --> SINGLE_MASK[截取若干帧生成对应模板]
    SINGLE_MASK -->|逻辑与运算|MASK[最终水印模板]
    MASK --> FRAME[读取视频]
    FRAME --> AUDIO[抽取音频] 
    FRAME --> INPAINT[TELEA算法逐帧修复]
    INPAINT --> VIDEO[逐帧写入视频]
    AUDIO --> OUTPUT
    VIDEO --> OUTPUT[合并封装输出视频]
    ```
  - **去除字幕**
    ```mermaid
    graph LR
    ROI[框选字幕] --> FRAME[读取视频]
    FRAME --> MASK[生成单帧图像模板]
    FRAME --> AUDIO[抽取音频]
    MASK --> INPAINT[TELEA算法逐帧修复]
    INPAINT --> VIDEO[逐帧写入视频]
    VIDEO --> OUTPUT[合并封装输出视频]
    AUDIO --> OUTPUT
    ```