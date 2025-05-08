# AetherTouch：基于手势的物联网控制系统

一个运行在树莓派上的智能手势控制系统，利用计算机视觉技术实现与物联网设备的无接触交互。

## 项目概述

本系统结合了 YOLOv5 物体检测（通过 ONNX）和 MediaPipe 手部跟踪技术，使得用户能够通过直观的手势来控制物联网设备。比如，用户可以通过简单地指向物体或使用不同的手势来控制 LED 灯和音乐播放。

## 功能特点

- **物体检测**：使用 YOLOv5 检测 LED 灯、蜂鸣器和控制按钮
- **手势识别**：跟踪手部位置并识别多种手势：
  - 指向手势以激活设备
  - 捏合手势控制亮度
  - 两指手势控制音量
  - 张开手掌打开所有设备
  - 握拳关闭所有设备
- **多线程处理**：分离物体检测和手部跟踪以提高性能
- **数据分析**：记录用户交互并生成可视化报告
- **模型优化**：可选的 ONNX Runtime 优化以提高推理速度

## 硬件要求

- 树莓派 4B（推荐）
- 兼容树莓派的摄像头模块
- 连接到 GPIO 引脚的 LED 灯（默认：G17）
- 连接到 GPIO 引脚的蜂鸣器（默认：G18）
- 可选的"teeth"组件（控制按钮）

## 软件要求

- Python 3.7+
- OpenCV
- NumPy
- MediaPipe
- RPi.GPIO
- 可选：用于模型优化的 ONNX Runtime
- 可选：用于数据可视化的 Pandas 和 Matplotlib

3. 下载 YOLOv5 ONNX 模型：

出于自定义的目的，用户需要自行准备 YOLOv5 ONNX 格式模型文件，并命名为`model.onnx`放置在项目根目录下。

以下是笔者建议的几个模型选择，用户可以根据设备性能和需求自行选择：
[YOLOv5-Lite-v1.5](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.5)
[YOLOv5 v7.0 5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx)
[YOLOv5 v7.0 5s-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s-seg.pt)

## 使用方法

### 基本用法

运行主应用程序：

```bash
python aether.py
```

### 高级选项

```bash
# 启用交互记录和分析
python aether.py --log

# 启用模型优化（需要onnxruntime）
python aether.py --optimize

# 同时启用两项功能
python aether.py --log --optimize
```

### 手势指南 （当然，也可以自己自定义）

- **指向**物体以激活它
- **捏合**（拇指和食指合拢）同时指向 LED 以控制亮度
- **两指**（食指和中指伸出）同时指向蜂鸣器以控制音量
- **张开手掌**（所有手指伸展）打开所有设备
- **握拳**（所有手指闭合）关闭所有设备

## 项目结构

- `aether.py`：结合所有组件的主应用程序
- `gesture_recognition.py`：增强型手势识别模块
- `interaction_tracker.py`：数据记录和可视化模块
- `led_controller.py`：LED 控制模块
- `music_player.py`：蜂鸣器控制和音乐播放模块
- `gpio_config.py`：GPIO 引脚配置
- `music.txt`：蜂鸣器音乐播放的音符

## 数据分析

使用`--log`选项运行时，系统会记录所有交互并生成可视化报告，包括：

- 设备使用分布（饼图）
- 手势使用分布（饼图）
- 交互时间线
- 摘要统计

报告保存在`reports`目录中。

## 模型训练

系统使用 YOLOv5 物体检测模型，该模型经过训练可以识别 LED 灯、蜂鸣器和控制按钮。以下笔者提供一个可以自己训练模型的方法：

1. 使用以下命令收集图像：

```bash
python img_collection.py
```

2. 标记图像并将其转换为 YOLOv5 格式：

```bash
python lablemetoyolo.py
```

3. 按照[YOLOv5 文档](https://github.com/ultralytics/yolov5)训练 YOLOv5 模型

4. 导出为 ONNX 格式并放置在项目目录中

## 开发

### 附加文件

- `image_detector.py`：在单张图像上测试检测
- `video_detector.py`：基本视频测试
- `threaded_detector.py`：多线程视频测试

## 致谢

- [YOLOv5](https://github.com/ultralytics/yolov5)提供物体检测技术
- [MediaPipe](https://mediapipe.dev/)提供手部跟踪技术
- [OpenCV](https://opencv.org/)提供图像处理功能
