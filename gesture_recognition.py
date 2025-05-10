import cv2
import mediapipe as mp
import numpy as np
import math

class GestureRecognizer:
    """
    Enhanced gesture recognition based on MediaPipe hand landmarks
    """
    def __init__(self):
        # Initialize MediaPipe hand solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Define gesture thresholds
        self.pinch_threshold = 0.05
        self.finger_open_threshold = 0.3
        
        # Gesture history for stability
        self.gesture_history = []
        self.history_length = 5
        
    def recognize_gestures(self, frame):
        """
        Recognize gestures from input frame
        
        Args:
            frame: Input BGR frame
            
        Returns:
            gesture: Detected gesture type
            landmarks: Hand landmarks
            annotated_frame: Frame with visualization
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = frame.shape
        
        # Process image
        results = self.hands.process(rgb_frame)
        
        # Default values
        gesture = "NONE"
        index_finger_tip = (0, 0)
        
        # Make a copy for visualization
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get fingertip coordinates
                thumb_tip = (hand_landmarks.landmark[4].x * image_width, 
                             hand_landmarks.landmark[4].y * image_height)
                index_tip = (hand_landmarks.landmark[8].x * image_width, 
                            hand_landmarks.landmark[8].y * image_height)
                middle_tip = (hand_landmarks.landmark[12].x * image_width, 
                             hand_landmarks.landmark[12].y * image_height)
                ring_tip = (hand_landmarks.landmark[16].x * image_width, 
                           hand_landmarks.landmark[16].y * image_height)
                pinky_tip = (hand_landmarks.landmark[20].x * image_width, 
                            hand_landmarks.landmark[20].y * image_height)
                
                # Get wrist position
                wrist = (hand_landmarks.landmark[0].x * image_width, 
                         hand_landmarks.landmark[0].y * image_height)
                
                # Calculate distances
                thumb_to_index_distance = self._calculate_distance(thumb_tip, index_tip)
                
                # Store index finger position for pointing
                index_finger_tip = index_tip
                
                # Check if fingers are open or closed
                is_thumb_open = self._is_finger_open(hand_landmarks, 0, image_width, image_height)
                is_index_open = self._is_finger_open(hand_landmarks, 1, image_width, image_height)
                is_middle_open = self._is_finger_open(hand_landmarks, 2, image_width, image_height)
                is_ring_open = self._is_finger_open(hand_landmarks, 3, image_width, image_height)
                is_pinky_open = self._is_finger_open(hand_landmarks, 4, image_width, image_height)
                
                # Recognize gestures
                if not any([is_thumb_open, is_index_open, is_middle_open, is_ring_open, is_pinky_open]):
                    gesture = "ALL_OFF"  
                elif thumb_to_index_distance < self.pinch_threshold * image_width:
                    gesture = "BRIGHTNESS_CONTROL"
                elif is_index_open and not is_middle_open and not is_ring_open and not is_pinky_open:
                    gesture = "POINTING"
                elif is_index_open and is_middle_open and not is_ring_open and not is_pinky_open:
                    gesture = "VOLUME_CONTROL"
                elif all([is_index_open, is_middle_open, is_ring_open, is_pinky_open]):
                    gesture = "ALL_ON"
                else:
                    gesture = "UNKNOWN"
                
                # Stabilize gestures using history
                self.gesture_history.append(gesture)
                if len(self.gesture_history) > self.history_length:
                    self.gesture_history.pop(0)
                
                # Find most common gesture in history
                if self.gesture_history:
                    gesture = max(set(self.gesture_history), key=self.gesture_history.count)
                
                # Display detected gesture
                cv2.putText(
                    annotated_frame, 
                    f"Gesture: {gesture}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
        
        return gesture, index_finger_tip, annotated_frame
    
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _is_finger_open(self, hand_landmarks, finger_idx, img_width, img_height):
        """
        Check if a finger is open
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            finger_idx: Index of the finger (0=thumb, 1=index, 2=middle, 3=ring, 4=pinky)
            img_width: Frame width
            img_height: Frame height
            
        Returns:
            bool: True if finger is open, False otherwise
        """
        if finger_idx == 0:  # Thumb
            tip_idx = 4
            pip_idx = 3
            mcp_idx = 2
        else:
            tip_idx = finger_idx * 4 + 4
            pip_idx = finger_idx * 4 + 3
            mcp_idx = finger_idx * 4 + 2
        
        # Get coordinates
        tip = np.array([
            hand_landmarks.landmark[tip_idx].x * img_width,
            hand_landmarks.landmark[tip_idx].y * img_height
        ])
        pip = np.array([
            hand_landmarks.landmark[pip_idx].x * img_width,
            hand_landmarks.landmark[pip_idx].y * img_height
        ])
        mcp = np.array([
            hand_landmarks.landmark[mcp_idx].x * img_width,
            hand_landmarks.landmark[mcp_idx].y * img_height
        ])
        
        # Calculate distances
        dist_tip_pip = np.linalg.norm(tip - pip)
        dist_pip_mcp = np.linalg.norm(pip - mcp)
        
        # If tip is further from PIP than MCP is from PIP, finger is likely extended
        if finger_idx == 0:  # Special case for thumb
            wrist = np.array([
                hand_landmarks.landmark[0].x * img_width,
                hand_landmarks.landmark[0].y * img_height
            ])
            return np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist)
        else:
            return dist_tip_pip > self.finger_open_threshold * dist_pip_mcp
