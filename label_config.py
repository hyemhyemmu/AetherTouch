"""
Gesture Configuration Module

Define gesture and object recognition ID constant configurations for easy customization
"""

# Object ID configurations (corresponding to YOLO model output)
OBJECT_IDS = {
    "LED": 0,          # LED light
    "BUZZER": 1,       # Buzzer
    "FIST": 2,         # Fist gesture
    "OPEN_PALM": 3,    # Open palm
    "PINCH": 4,        # Pinch gesture
    "POINTING": 5,     # Pointing gesture
    "TWO_FINGERS": 6   # Two fingers gesture
}

# Label names for display purposes
LABEL_NAMES = {
    0: 'GPIO LED', 
    1: 'buzzer', 
    2: 'fist', 
    3: 'open_palm', 
    4: 'pinch', 
    5: 'pointing', 
    6: 'two_fingers'
}

# Gesture action configurations
GESTURE_ACTIONS = {
    "NONE": "NONE",                  # No gesture
    "POINTING": "POINTING",          # Pointing gesture
    "BRIGHTNESS_CONTROL": "BRIGHTNESS_CONTROL",  # Brightness control
    "VOLUME_CONTROL": "VOLUME_CONTROL",          # Volume control
    "ALL_ON": "ALL_ON",              # Turn all devices on
    "ALL_OFF": "ALL_OFF"             # Turn all devices off
}

# Device types
DEVICE_TYPES = {
    "LED": "LED",
    "BUZZER": "BUZZER",
    "ALL": "ALL"
}

# Action types
ACTION_TYPES = {
    "ON": "ON",
    "OFF": "OFF",
    "BRIGHTNESS": "BRIGHTNESS",
    "VOLUME": "VOLUME"
}

# For backward compatibility with tools/labelmetoyolo.py
# Class labels mapped to their corresponding index values
class_labels = {
    'led': 0,       # LED light component
    'buzzer': 1,    # Buzzer/speaker component
    'fist': 2,      # Fist gesture
    'open_palm': 3, # Open palm gesture
    'pinch': 4,     # Pinch gesture
    'pointing': 5,  # Pointing gesture
    'two_fingers': 6  # Two fingers gesture
}

# Dataset configuration parameters
dataset_config = {
    'path_json': 'labs',  # Path to JSON label files
    'train_ratio': 0.9    # Train/test split ratio
}