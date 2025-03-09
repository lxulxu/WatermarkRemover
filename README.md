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

- 克隆仓库

```bash
git clone https://github.com/lxulxu/WatermarkRemover.git
cd WatermarkRemover
```

- 创建并激活虚拟环境（可选，推荐）

```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

- 安装基础依赖

```bash
pip install -r requirements.txt
```

- 安装PyTorch（二选一）

  1. CPU版本

    ```bash
    pip install torch
    ```
  
  2. GPU版本（需要NVIDIA显卡）

      - 安装CUDA Toolkit

  	    访问 [NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-downloads)，选择对应的操作系统和版本。

      - 安装cuDNN
  
        访问 [NVIDIA cuDNN下载页面](https://developer.nvidia.com/cudnn-downloads)，选择与CUDA版本匹配的cuDNN。

      - 安装GPU版本的PyTorch

        访问 [PyTorch官方网站](https://pytorch.org/get-started/locally/)，选择与CUDA版本匹配的命令安装，例如：
        
  
         ```bash
          pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
         ```
  


​	程序会自动检测是否有可用的GPU输出相关信息并自动选择处理方式。

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

1. **水印区域选择**：程序会显示视频一帧，手动框选水印区域后按**SPACE**或**ENTER**键继续。
2. **效果预览**（可选）：显示处理效果预览，按**SPACE**或**ENTER**键确认或按**ESC**键取消退出程序。
3. **视频处理**：初次运行程序使用LAMA模型需较长时间下载模型。
4. **输出结果**

## 局限性

- 只能处理固定位置的水印（不支持移动水印）
- 同一批处理的视频尺寸必须一致
- 同一批处理的视频水印必须一致

## 常见问题

**Q: 安装CUDA后出现错误？**
 A: 确保安装的CUDA、cuDNN和PyTorch版本相互兼容。参考PyTorch官方网站的兼容性表格。



## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lxulxu/WatermarkRemover&type=Date)](https://star-history.com/#lxulxu/WatermarkRemover&Date)
