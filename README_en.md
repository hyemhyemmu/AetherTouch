# AetherTouch: YOLO-Based Gesture IoT Control System


> Returning interaction to instinct, redefining communication with smart devices through gestures

An innovative gesture control system running on Raspberry Pi, integrating cutting-edge computer vision technology to achieve natural, contactless human-computer interaction with IoT devices.

## Project Highlights

## Project Overview

The AetherTouch system cleverly combines YOLOv5 object detection (optimized through ONNX) and MediaPipe hand tracking technology, enabling users to control surrounding smart devices through intuitive and natural gestures. The system not only recognizes the user's pointing intent but also parses complex gestures to achieve fine-grained operations such as brightness adjustment and volume control, providing a new paradigm for IoT device interaction.

This project stems from a simple belief: technology should integrate into our lives more naturally and intuitively. By eliminating the limitations of physical remote controls and switches, AetherTouch allows users to communicate with the surrounding smart environment with just a wave of their hand.

## Features

- **Smart Object Detection**: Uses an optimized YOLOv5 model to detect and locate IoT devices such as LED lights, buzzers, and control buttons in real-time
- **High-Precision Gesture Recognition**: Innovatively applies MediaPipe tracking technology to achieve precise recognition of various gestures:
  - Pointing gesture: Activates specific devices (spatial positioning accuracy of ±2cm)
  - Pinch gesture: Continuously adjusts LED brightness (supports fine control from 0-100%)
  - Two-finger gesture: Controls volume and tone
  - Open palm: Activates all components
  - Fist gesture: Turns off all components
- **Efficient Multi-threaded Architecture**: Separates object detection and hand tracking into independent threads, significantly improving system response speed (average latency reduced by 47%)
- **Data-Driven Analysis**: Built-in user behavior analysis engine optimizes interaction experience through visualization reports
- **Edge Computing Optimization**: ONNX Runtime optimization designed for resource-constrained devices, achieving stable performance of 15FPS on average on Raspberry Pi

## Hardware Configuration

- Raspberry Pi 4B
- Raspberry Pi compatible CSI/USB camera module (recommended 720p@30fps or above)
- LED array connected to GPIO (default: G17, expandable)
- Buzzer module connected to GPIO (default: G18, supports PWM control)
- Optional "teeth" auxiliary control button

## Software Configuration

- Python 3.7+
- OpenCV
- NumPy
- MediaPipe
- RPi.GPIO
- Optional: ONNX Runtime for model optimization
- Optional: Pandas and Matplotlib for data visualization

## Usage

### Basic Usage

Run the main application:

```bash
python aether.py
```

### Advanced Options

```bash
# Enable interaction logging and analysis
python aether.py --log

# Enable model optimization (requires onnxruntime)
python aether.py --optimize

# Enable both features
python aether.py --log --optimize
```

### Gesture Guide (of course, you can customize)

- **Point** at an object to activate it
- **Pinch** (thumb and index finger together) while pointing at an LED to control brightness
- **Two fingers** (index and middle fingers extended) while pointing at a buzzer to control volume
- **Open palm** to turn on all components
- **Fist** to turn off all components

## Project Structure

- `aether.py`: Main application combining all components
- `gesture_recognition.py`: Enhanced gesture recognition module
- `interaction_tracker.py`: Data logging and visualization module
- `led_controller.py`: LED control module
- `music_player.py`: Buzzer control and music playback module
- `gpio_config.py`: GPIO pin configuration
- `music.txt`: Notes for buzzer music playback

## Data Analysis

When running with the `--log` option, the system records all interactions and generates visualization reports, including:

- Device usage distribution (pie chart)
- Gesture usage distribution (pie chart)
- Interaction timeline
- Interaction performance metrics (response time, recognition accuracy, etc.)

Reports are saved in the `reports` directory.

In our small user study (n=11), participants mastered all gestures in an average of 2.5 minutes, and the system achieved a recognition accuracy of 92.7%, far exceeding the learning efficiency of traditional button-based interactions.

## Model Details

The core of the project is a carefully tuned YOLOv5s model that achieves the optimal balance between speed and accuracy, especially suitable for deployment on edge computing devices. The model was trained with input images of **640x640 pixels**, which has been proven to be the optimal configuration through multiple experiments.

The system uses a data-driven approach: I personally collected and annotated a dataset of over **1100 images** using RoboFlow, covering **7 key categories**. Various data augmentation techniques were applied during training, including random rotation, brightness changes, and perspective transformation, enabling the model to work stably under various lighting conditions.

Here are the core performance metrics of the model on the validation set:

- **Overall Precision:** 0.783
- **Overall Recall:** 0.920
- **mAP@0.5:** 0.868
- **mAP@0.5:0.95:** 0.685

To achieve real-time performance, I used the ONNX format to quantize and optimize the model, reducing inference latency by approximately 35% while maintaining recognition accuracy.

Detailed training parameter configuration, optimization strategies, and complete experimental data can be found in the [`para.md`](./para.md) file.

## Custom Model:

The design philosophy of AetherTouch is openness and customizability. The instructions and components in the current project are just a starting point. Here's how to extend the system according to your needs:

Here are some recommended model choices that you can select based on device performance and requirements:

- [YOLOv5-Lite-v1.5](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.5)
- [YOLOv5 v7.0 5s](https://github.com/ultralytics/yolov5/releases/download/v7.0/)
- [YOLOv5 v7.0 5s-seg](https://github.com/ultralytics/yolov5/releases/download/v7.0/)

Here's a method for training your own model:

1. Collect images:

```bash
python tools/img_collection.py
```

Add gestures or components as needed

2. Label images (e.g., using labelImg, RoboFlow) and convert them to YOLOv5 format:

```bash
python tools/lablemetoyolo.py
```

3. Train a YOLOv5 model following the [YOLOv5 documentation](https://github.com/ultralytics/yolov5)

4. Export to ONNX format and place it in the project root directory, named `model.onnx`

### Additional Files

- `image_detector.py`: Test detection on a single image
- `video_detector.py`: Basic video testing
- `threaded_detector.py`: Multi-threaded video testing

## Motivation and Future Outlook

The inspiration for this project comes from thinking about gestures as a mode of human-computer interaction. I believe that gestures, as one of the most primitive and natural forms of communication for humans, will play an increasingly important role in the field of human-computer interaction. Their contactless nature and low learning cost make them suitable for a wide range of application scenarios and user groups, such as smart homes, contactless medical environments, and various industrial application scenarios.

While implementing complex, intelligent, and natural human-computer interaction on resource-limited edge devices is challenging, it is also very promising, especially with the emergence of efficient technologies like YOLO. Gesture recognition is just the first step; in the future, we may try to integrate voice control and other methods to provide more interaction options and create a multi-modal interaction system. At the same time, the improvement of edge computing capabilities may allow for more complex gesture combinations and even intelligent decision-making in combination with contextual environments.

## Conclusion

AetherTouch is my passionate attempt to explore the field of human-computer interaction. From an initial simple idea, to step-by-step overcoming technical challenges, and finally implementing a usable prototype system, this process has been full of learning joy and creative delight. Through continuous adjustment and optimization, I ultimately achieved a smooth interaction experience under hardware constraints, and this process of balancing technology and user experience has been tremendously beneficial.

I hope this project can inspire more people to think about and explore natural human-computer interaction. Whether as a learning reference or as a foundation for practical applications, AetherTouch will continue to evolve, embracing the broad prospects of future smart homes and the Internet of Things.

## Acknowledgements

Thanks to the valuable tools from the open-source community:

- [YOLOv5](https://github.com/ultralytics/yolov5) for providing object detection technology
- [MediaPipe](https://mediapipe.dev/) for providing hand tracking technology
- [OpenCV](https://opencv.org/) for providing image processing functionality
