import cv2
from threading import Thread
import uuid
import os
import time

# Global variables
count = 0
current_category = "default"
is_collecting = False
categories = [
    "led",          # LED light
    "buzzer",       # Buzzer
    "pointing",     # Pointing gesture
    "pinch",        # Pinch gesture
    "two_fingers",  # Two-finger gesture
    "open_palm",    # Open palm
    "fist",         # Fist
]

def image_collect(cap):
    global count, is_collecting
    folder_path = os.path.join('images', current_category)
    
    while is_collecting:
        success, img = cap.read()
        if success:
            file_name = f"{current_category}_{count:04d}_{uuid.uuid4()}.jpg"
            cv2.imwrite(os.path.join(folder_path, file_name), img)
            count += 1
            print(f"Saved [{current_category}] {count} {file_name}")
        time.sleep(0.4)

def select_category():
    global current_category
    
    print("\nPlease select a category to collect:")
    for i, category in enumerate(categories):
        print(f"{i+1}. {category}")
    
    while True:
        try:
            choice = int(input(f"\nEnter category number (1-{len(categories)}): "))
            if 1 <= choice <= len(categories):
                current_category = categories[choice-1]
                break
            else:
                print(f"Please enter a number between 1 and {len(categories)}")
        except ValueError:
            print("Please enter a valid number")
    
    # Create folder for the selected category
    folder_path = os.path.join('images', current_category)
    os.makedirs(folder_path, exist_ok=True)
    
    # Reset the counter for this category
    global count
    count = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\nSelected: {current_category}")
    print(f"Images will be saved to: {folder_path}")
    print(f"The folder already has {count} images")
    print("\nInstructions:")
    print("- Press 'c' to start collection")
    print("- Press 's' to stop collection")
    print("- Press 'n' to select a new category")
    print("- Press 'q' to exit program")

if __name__ == "__main__":
    # Create main image folder
    os.makedirs("images", exist_ok=True)
    
    # Initial category selection
    select_category()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        exit()
    
    # Create thread but don't start immediately
    m_thread = None
    
    while True:
        # Read a frame
        success, img = cap.read()
        
        if not success:
            print("Warning: Failed to capture frame")
            continue
        
        # Display current category and count
        status_text = f"{current_category}: {count}"
        cv2.putText(img, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display collection status
        if is_collecting:
            cv2.putText(img, "Collecting", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # Display image
        cv2.imshow("AetherTouch Data Collection", img)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Key controls
        if key == ord('c') and not is_collecting:
            # Start collection
            is_collecting = True
            m_thread = Thread(target=image_collect, args=([cap]), daemon=True)
            m_thread.start()
            print(f"Started collecting [{current_category}]")
            
        elif key == ord('s') and is_collecting:
            # Stop collection
            is_collecting = False
            print(f"Stopped collecting [{current_category}], collected {count} images")
            
        elif key == ord('n'):
            # If collecting, stop first
            if is_collecting:
                is_collecting = False
                print(f"Collection stopped. Collected {count} [{current_category}] images")
            
            # Select new category
            select_category()
            
        elif key == ord('q'):
            # Exit program
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nData collection completed")
    print("Image statistics by category:")
    
    total = 0
    for category in categories:
        folder_path = os.path.join('images', category)
        if os.path.exists(folder_path):
            count = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"- {category}: {count} images")
            total += count
    
    print(f"Total: {total} images")
    