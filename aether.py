import cv2  # Note: requires opencv-python package
import numpy as np
import time
from threading import Thread
import mediapipe as mp  # Note: requires mediapipe package
from gpio_config import gpio_pins
from led_controller import LEDController
import threading
from music_player import MusicPlayer
import argparse
import os
import random

# ----- PARAMETERS/VARIABLES ------

# camera resolution
CAMERA_WIDTH = 480
CAMERA_HEIGHT = 640

# YOLO model input resolution
# standard YOLO model usually use 640x640 input
MODEL_WIDTH = 640
MODEL_HEIGHT = 640

# Music file path
MUSIC_FILE = 'music.txt'

# ---------------------------------

# Import newly added modules
from gesture_recognition import GestureRecognizer
from interaction_tracker import InteractionTracker

# Global variables for sharing data between threads
det_boxes_show = []
scores_show = []
ids_show = []
FPS_show = ""
brightness_level = 100
volume_level = 100

# Define LED object
m_light = LEDController(gpio_pins["G17"])

# Define buzzer object
m_runing_song = MusicPlayer(gpio_pins['G18'])

# The pretrained model file (users need to provide their own)
DEFAULT_MODEL = "model.onnx"

# ------------------------

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def post_process_opencv(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
    """
    Process YOLO model outputs with OpenCV
    
    Args:
        outputs: Raw model outputs
        model_h, model_w: Model input dimensions
        img_h, img_w: Original image dimensions
        thred_nms: NMS threshold
        thred_cond: Confidence threshold
    
    Returns:
        det_boxes: Detection bounding boxes
        scores: Detection confidence scores
        ids: Class IDs
    """
    conf = outputs[:,4].tolist()
    c_x = outputs[:,0]/model_w*img_w
    c_y = outputs[:,1]/model_h*img_h
    w  = outputs[:,2]/model_w*img_w
    h  = outputs[:,3]/model_h*img_h
    p_cls = outputs[:,5:]
    if len(p_cls.shape)==1:
        p_cls = np.expand_dims(p_cls,1)
    cls_id = np.argmax(p_cls,axis=1)

    p_x1 = np.expand_dims(c_x-w/2,-1)
    p_y1 = np.expand_dims(c_y-h/2,-1)
    p_x2 = np.expand_dims(c_x+w/2,-1)
    p_y2 = np.expand_dims(c_y+h/2,-1)
    areas = np.concatenate((p_x1,p_y1,p_x2,p_y2),axis=-1)
    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
    if len(ids)>0:
        return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
    else:
        return [], [], []

def infer_image(net, img0, model_h, model_w, thred_nms=0.4, thred_cond=0.5):
    """
    Run inference on an image using ONNX model
    
    Args:
        net: OpenCV DNN network
        img0: Input image
        model_h, model_w: Model input dimensions
        thred_nms: NMS threshold
        thred_cond: Confidence threshold
    
    Returns:
        det_boxes: Detection bounding boxes
        scores: Detection confidence scores
        ids: Class IDs
    """
    img = img0.copy()
    img = cv2.resize(img, [model_h, model_w])
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, swapRB=True)
    net.setInput(blob)
    outs = net.forward()[0]
    
    det_boxes, scores, ids = post_process_opencv(
        outs, model_h, model_w, img0.shape[0], img0.shape[1], thred_nms, thred_cond
    )
    return det_boxes, scores, ids

def optimize_model(model_path):
    """
    Optimize ONNX model for better performance
    
    Args:
        model_path: Path to ONNX model
        
    Returns:
        Path to optimized model
    """
    try:
        import onnxruntime as ort
        print("Optimizing model with ONNX Runtime...")
        
        # Configure optimization options
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.optimized_model_filepath = "optimized_model.onnx"
        
        # Create optimized model
        _ = ort.InferenceSession(model_path, options)
        print(f"Model optimized and saved to: {options.optimized_model_filepath}")
        
        return options.optimized_model_filepath
    
    except ImportError:
        print("ONNX Runtime not available. Using original model.")
        return model_path
    except Exception as e:
        print(f"Error optimizing model: {e}")
        return model_path

def m_detection(net, cap, model_h, model_w):
    """
    Object detection thread function
    
    Args:
        net: OpenCV DNN network
        cap: Video capture device
        model_h, model_w: Model input dimensions
    """
    global det_boxes_show
    global scores_show
    global ids_show
    global FPS_show
    
    while True:
        success, img0 = cap.read()
        if success:
            t1 = time.time()
            det_boxes, scores, ids = infer_image(
                net, img0, model_h, model_w, thred_nms=0.4, thred_cond=0.4
            )
            t2 = time.time()
            str_fps = f"FPS: {1./(t2-t1):.2f}"

            det_boxes_show = det_boxes
            scores_show = scores
            ids_show = ids
            FPS_show = str_fps

def light_on(level=None):
    """Turn on LED with optional brightness level"""
    global m_light
    global brightness_level
    
    if level is not None:
        brightness_level = level
        
    m_light.turn_on()
    print(f"Light turned on (brightness: {brightness_level}%)")
    return "LED", "ON", brightness_level

def light_off():
    """Turn off LED"""
    global m_light
    m_light.turn_off()
    print("Light turned off")
    return "LED", "OFF", 0

def set_brightness(level):
    """Set LED brightness level"""
    global brightness_level
    
    # Clamp level between 0-100%
    brightness_level = max(0, min(100, level))
    
    # Implement with PWM if hardware supports it
    # For now, just turn on/off based on threshold
    if brightness_level > 0:
        light_on(brightness_level)
    else:
        light_off()
    
    print(f"Brightness set to {brightness_level}%")
    return "LED", "BRIGHTNESS", brightness_level

def music_on(volume=None):
    """Turn on music with optional volume level"""
    global m_runing_song
    global volume_level
    
    if volume is not None:
        volume_level = volume
    
    if m_runing_song.isAlive() == False:
        # Create and start thread if not running
        m_runing_song = MusicPlayer(gpio_pins['G18'])
        m_runing_song.load_music_file(MUSIC_FILE)
        m_runing_song.setDaemon(True)
        m_runing_song.start()
        
    else: 
        # Restart if already playing
        m_runing_song.stop_playback()
        time.sleep(0.1)
        m_runing_song.join()

        # Reload and start
        m_runing_song = MusicPlayer(gpio_pins['G18'])
        m_runing_song.load_music_file(MUSIC_FILE)
        m_runing_song.setDaemon(True)
        m_runing_song.start()
    
    print(f"Music started (volume: {volume_level}%)")
    return "BUZZER", "ON", volume_level
    
def music_off():
    """Turn off music"""
    global m_runing_song
    
    if m_runing_song.isAlive() == True:
        m_runing_song.stop_playback()
        time.sleep(0.1)
        m_runing_song.join()  
    
    print("Music stopped")
    return "BUZZER", "OFF", 0

def set_volume(level):
    """Set music volume level"""
    global volume_level
    
    # Clamp level between 0-100%
    volume_level = max(0, min(100, level))
    
    # Implement volume control if hardware supports it
    # For now, just turn on/off based on threshold
    if volume_level > 0:
        music_on(volume_level)
    else:
        music_off()
    
    print(f"Volume set to {volume_level}%")
    return "BUZZER", "VOLUME", volume_level

def all_off():
    """Turn off all devices"""
    device1, action1, level1 = light_off()
    device2, action2, level2 = music_off()
    
    print("All devices turned off")
    return [
        (device1, action1, level1),
        (device2, action2, level2)
    ]

def check_point_in_box(point, box):
    """Check if a point is inside a bounding box"""
    x, y = point
    x1, y1, x2, y2 = box
    return (x > x1 and x < x2 and y > y1 and y < y2)

def apply_gesture_action(gesture, pos, boxes, ids, scores, labels, logger=None):
    """
    Apply action based on detected gesture and object
    
    Args:
        gesture: Detected gesture type
        pos: Finger position (x, y)
        boxes: Detection boxes
        ids: Class IDs
        scores: Confidence scores
        labels: Class labels
        logger: Interaction logger for recording actions
        
    Returns:
        action_performed: Whether an action was performed
    """
    action_performed = False
    
    # Process based on gesture type
    if gesture == "POINTING":
        # Check if pointing at any object
        for box, score, id in zip(boxes, scores, ids):
            if check_point_in_box(pos, box):
                if id == 0:  # LED
                    device, action, level = light_on()
                    action_performed = True
                elif id == 1:  # Buzzer
                    device, action, level = music_on()
                    action_performed = True
                elif id == 2:  # Teeth (off button)
                    actions = all_off()
                    action_performed = True
                    device, action, level = "ALL", "OFF", 0
                
                # Log the interaction
                if logger and action_performed:
                    logger.record_event(action, device, gesture, 0)
                    
    elif gesture == "BRIGHTNESS_CONTROL":
        # Find LED in detected objects
        for box, score, id in zip(boxes, scores, ids):
            if id == 0:  # LED
                # Calculate brightness based on vertical position
                # Lower position = brighter
                _, screen_height = CAMERA_WIDTH, CAMERA_HEIGHT  
                rel_y_pos = max(0, min(1, pos[1] / screen_height))
                brightness = int((1 - rel_y_pos) * 100)  # Invert: lower = brighter
                
                device, action, level = set_brightness(brightness)
                action_performed = True
                
                # Log the interaction
                if logger:
                    logger.record_event(action, device, gesture, 0)
                break
                
    elif gesture == "VOLUME_CONTROL":
        # Find buzzer in detected objects
        for box, score, id in zip(boxes, scores, ids):
            if id == 1:  # Buzzer
                # Calculate volume based on vertical position
                # Lower position = louder
                _, screen_height = CAMERA_WIDTH, CAMERA_HEIGHT  
                rel_y_pos = max(0, min(1, pos[1] / screen_height))
                volume = int((1 - rel_y_pos) * 100)  # Invert: lower = louder
                
                device, action, level = set_volume(volume)
                action_performed = True
                
                # Log the interaction
                if logger:
                    logger.record_event(action, device, gesture, 0)
                break
                
    elif gesture == "ALL_ON":
        # Turn on all devices
        light_on()
        music_on()
        action_performed = True
        
        # Log the interaction
        if logger:
            logger.record_event("ON", "ALL", gesture, 0)
            
    elif gesture == "ALL_OFF":
        # Turn off all devices
        all_off()
        action_performed = True
        
        # Log the interaction
        if logger:
            logger.record_event("OFF", "ALL", gesture, 0)
    
    return action_performed


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Raspberry Pi Gesture Control System')
    parser.add_argument('--optimize', action='store_true', help='Optimize model for inference')
    parser.add_argument('--log', action='store_true', help='Enable interaction logging')
    parser.add_argument('--model-path', type=str, help='Custom path to ONNX model file')
    args = parser.parse_args()
    
    # Initialize the enhanced gesture recognizer
    gesture_recognizer = GestureRecognizer()
    
    # Initialize interaction logger if enabled
    interaction_logger = None
    if args.log:
        interaction_logger = InteractionTracker()
    
    # YOLO model loading
    dic_labels = {0:'led', 1:'buzzer', 2:'teeth'}
    model_h = MODEL_HEIGHT
    model_w = MODEL_WIDTH
    
    # Set model path
    file_model = args.model_path if args.model_path else DEFAULT_MODEL
    print(f"Using model: {file_model}")
    
    # Check if model file exists
    if not os.path.exists(file_model):
        print(f"Error: Model file {file_model} not found.")
        print("Please download a YOLOv5 model and convert it to ONNX format.")
        print("Then place it in the project directory as 'model.onnx'")
        print("You can download YOLOv5 models from https://github.com/ultralytics/yolov5/releases")
        exit(1)
    
    # Optimize model if requested
    if args.optimize:
        file_model = optimize_model(file_model)
    
    try:
        net = cv2.dnn.readNet(file_model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model file exists and is a valid ONNX model.")
        exit(1)
 
    # Initialize video capture
    video = 0
    cap = cv2.VideoCapture(video)
    
    # Start object detection thread
    m_thread = Thread(target=m_detection, args=([net, cap, model_h, model_w]), daemon=True)
    m_thread.start()
    
    print("System initialized. Press 'q' to exit.")
    
    
    while True:
        success, img0 = cap.read()
        if success:
            # Hand gesture detection
            gesture, finger_pos, annotated_frame = gesture_recognizer.recognize_gestures(img0)
            
            # Object detection visualization
            for box, score, id in zip(det_boxes_show, scores_show, ids_show):
                label = f'{dic_labels[id]}:{score:.2f}'
                plot_one_box(box, annotated_frame, color=(255,0,0), label=label, line_thickness=None)
            
            # Process gestures and interactions
            action_taken = apply_gesture_action(
                gesture, 
                finger_pos, 
                det_boxes_show, 
                ids_show, 
                scores_show, 
                dic_labels,
                interaction_logger
            )
            
            # Add FPS information
            cv2.putText(
                annotated_frame,
                FPS_show,
                (50, 80),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                3
            )
            
            # Display the frame
            cv2.imshow("Gesture Control System", annotated_frame)
            
        # Check for exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Generate interaction report if logging was enabled
    if interaction_logger:
        report_dir = interaction_logger.create_analysis_report()
        print(f"Interaction report saved to: {report_dir}")

