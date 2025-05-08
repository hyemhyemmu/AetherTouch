"""
Label Configuration Module

Defines labels and configuration parameters for object detection
and dataset management in the Raspberry Pi computer vision system.
"""

# Class labels mapped to their corresponding index values
# Used for object detection model training and inference
class_labels = {
    'led': 0,       # LED light component
    'buzzer': 1,    # Buzzer/speaker component
    'teeth': 2      # Teeth reference object
}

# Dataset configuration parameters
dataset_config = {
    'path_json': 'labs',  # Path to JSON label files
    'train_ratio': 0.9    # Train/test split ratio
}

# For backward compatibility
dic_labels = {**class_labels, 'path_json': dataset_config['path_json'], 'ratio': dataset_config['train_ratio']}