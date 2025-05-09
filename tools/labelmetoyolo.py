"""
LabelMe to YOLO Format Converter

This module converts annotation files from LabelMe JSON format to YOLO format.
It also extracts the embedded image data from the JSON files and saves them
as separate image files.

The converter automatically splits the dataset into training and validation sets
based on a configurable ratio.
"""
import os
import json
import random
import base64
import shutil
import argparse
from pathlib import Path
from glob import glob
from label_config import class_labels, dataset_config

def convert_labelme_to_yolo(config):
    """
    Convert LabelMe JSON annotations to YOLO format
    
    Args:
        config: Dictionary containing configuration parameters
                - path_json: Directory containing LabelMe JSON files
                - train_ratio: Ratio of data to use for training vs. validation
    
    Returns:
        None - Files are written directly to train and valid directories
    """
    path_input_json = config['path_json']
    train_ratio = config['train_ratio']
    
    # Process each JSON file
    for index, labelme_annotation_path in enumerate(glob(f'{path_input_json}/*.json')):

        # Get filename without extension
        image_id = os.path.basename(labelme_annotation_path).rstrip('.json')
        
        # Determine if sample goes to training or validation set
        dataset_type = 'train' if random.random() < train_ratio else 'valid'

        # Read LabelMe JSON annotation file
        with open(labelme_annotation_path, 'r') as labelme_annotation_file:
            labelme_annotation = json.load(labelme_annotation_file)

        # Create YOLO format annotation file
        yolo_annotation_path = os.path.join(dataset_type, 'labels', image_id + '.txt')
        
        # Extract and save image data from JSON
        yolo_image_path = os.path.join(dataset_type, 'images', image_id + '.jpg')
        
        try:
            # Decode base64 image data and save to file
            yolo_image = base64.decodebytes(labelme_annotation['imageData'].encode())
            with open(yolo_image_path, 'wb') as yolo_image_file:
                yolo_image_file.write(yolo_image)
        except KeyError:
            print(f"Warning: No imageData found in {labelme_annotation_path}")
            continue
        except Exception as e:
            print(f"Error extracting image from {labelme_annotation_path}: {e}")
            continue

        # Create annotation file
        with open(yolo_annotation_path, 'w') as yolo_annotation_file:
            # Process each shape/annotation
            for shape in labelme_annotation['shapes']:
                # YOLO only supports rectangle annotations
                if shape['shape_type'] != 'rectangle':
                    print(f"Skipping non-rectangle annotation in {labelme_annotation_path}")
                    continue
                
                # Extract coordinates and convert to YOLO format
                # YOLO format: class_id x_center y_center width height
                # (all normalized to 0-1 range)
                points = shape['points']
                
                # Get image dimensions
                img_width = labelme_annotation['imageWidth']
                img_height = labelme_annotation['imageHeight']
                
                # Calculate normalized values
                scale_width = 1.0 / img_width
                scale_height = 1.0 / img_height
                
                # Calculate width and height
                width = (points[1][0] - points[0][0]) * scale_width
                height = (points[1][1] - points[0][1]) * scale_height
                
                # Calculate center coordinates
                x_center = ((points[1][0] + points[0][0]) / 2) * scale_width
                y_center = ((points[1][1] + points[0][1]) / 2) * scale_height
                
                # Get class ID
                try:
                    object_class = class_labels[shape['label']]
                except KeyError:
                    print(f"Warning: Label '{shape['label']}' not found in class_labels")
                    continue
                
                # Write to YOLO format file
                yolo_annotation_file.write(f"{object_class} {x_center} {y_center} {width} {height}\n")
        
        print(f"Processed {index+1}: {image_id} -> {dataset_type}")


if __name__ == "__main__":
    # Create directory structure for YOLO format dataset
    os.makedirs(os.path.join("train", 'images'), exist_ok=True)
    os.makedirs(os.path.join("train", 'labels'), exist_ok=True)
    os.makedirs(os.path.join("valid", 'images'), exist_ok=True)
    os.makedirs(os.path.join("valid", 'labels'), exist_ok=True)
    
    # Create configuration dictionary
    conversion_config = {
        'path_json': dataset_config['path_json'],
        'train_ratio': dataset_config['train_ratio']
    }
    
    # Run the conversion
    print(f"Starting conversion from LabelMe to YOLO format...")
    print(f"JSON source directory: {conversion_config['path_json']}")
    print(f"Train/Validation split: {conversion_config['train_ratio'] * 100}% / {(1 - conversion_config['train_ratio']) * 100}%")
    
    convert_labelme_to_yolo(conversion_config)
    
    print("Conversion complete!")

