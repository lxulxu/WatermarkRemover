# WatermarkRemover

一个基于LAMA模型的视频水印移除工具，能够批量清除视频中的固定水印。

## 效果展示

原始帧
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/origin.jpg'>

去除水印
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/no_watermark.jpg'>

## 系统要求

- Python 3.10

## 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/lxulxu/WatermarkRemover.git
cd WatermarkRemover
```

1. 创建并激活虚拟环境（推荐）

```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

1. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

处理单个视频目录中的所有视频：

```bash
python watermark_remover.py --input /path/to/videos --output /path/to/output
```

### 带预览的处理

```bash
python watermark_remover.py --input /path/to/videos --output /path/to/output --preview
```

### 命令行参数

| 参数        | 简写 | 说明                   | 默认值         |
| ----------- | ---- | ---------------------- | -------------- |
| `--input`   | `-i` | 包含视频文件的输入目录 | `.` (当前目录) |
| `--output`  | `-o` | 处理后视频的输出目录   | `output`       |
| `--preview` | `-p` | 启用处理效果预览       | 禁用           |

## 工作流程

1. **水印区域选择**：程序会显示视频的第一帧，手动框选水印区域后按**SPACE**或**ENTER**键继续。
2. **效果预览**（可选）：显示处理效果预览，用户可按**SPACE**或**ENTER**键确认或按**ESC**键取消退出程序。
3. **视频处理**：初次运行程序使用LAMA模型需较长时间下载模型。
4. **输出结果**

## 性能说明

- 处理一个640x320的5秒视频大约需要14秒（处理速度会受到视频分辨率、CPU性能和所选区域大小的影响）

## 局限性

- 只能处理固定位置的水印（不支持移动水印）
- 同一批处理的视频尺寸必须一致
- 同一批处理的视频水印必须一致

## 高级配置

程序内部包含一些可调整的参数：

- `cache_size`：帧缓存大小，影响内存使用和处理速度
- `similarity_threshold`：相似帧判定阈值
- `keyframe_interval`：关键帧间隔，影响处理质量和速度
- `scene_change_threshold`：场景变化检测阈值

这些参数可以在代码中调整以适应不同的使用场景。

## 常见问题

**Q: 程序处理速度很慢，如何加快？**
 A: 尝试减小水印区域大小，增加缓存大小和关键帧间隔。

**Q: 处理后的视频有闪烁现象？**
 A: 尝试降低场景变化阈值和增加关键帧频率。

**Q: 水印没有完全去除？**
 A: 尝试在选择区域时包含更大的水印边缘区域，或调整水印掩码生成的参数。

