import cv2
import numpy as np
import time
import random 
from threading import Thread, Lock

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
    Process detection outputs from the model
    
    Args:
        outputs: Model output tensor
        model_h, model_w: Model input dimensions
        img_h, img_w: Original image dimensions
        thred_nms: NMS threshold
        thred_cond: Confidence threshold
        
    Returns:
        boxes, confidence scores, and class ids
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
    if len(ids) > 0:
        return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
    else:
        return [], [], []

def infer_image(net, img0, model_h, model_w, thred_nms=0.4, thred_cond=0.5):
    """
    Run inference on a single image
    
    Args:
        net: OpenCV DNN network
        img0: Input image
        model_h, model_w: Model dimensions
        thred_nms: NMS threshold
        thred_cond: Confidence threshold
        
    Returns:
        Detection boxes, scores, and class IDs
    """
    img = img0.copy()
    img = cv2.resize(img, [model_h, model_w])
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, swapRB=True)
    net.setInput(blob)
    outs = net.forward()[0]
    
    det_boxes, scores, ids = post_process_opencv(outs, model_h, model_w, 
                                                img0.shape[0], img0.shape[1], 
                                                thred_nms, thred_cond)
    return det_boxes, scores, ids

# Thread-safe detection results
class DetectionResults:
    def __init__(self):
        self.lock = Lock()
        self.boxes = []
        self.scores = []
        self.ids = []
        self.fps = ""
        
    def update(self, boxes, scores, ids, fps):
        with self.lock:
            self.boxes = boxes
            self.scores = scores
            self.ids = ids
            self.fps = fps
            
    def get(self):
        with self.lock:
            return self.boxes.copy(), self.scores.copy(), self.ids.copy(), self.fps

# Detection thread function
def detection_thread(net, cap, model_h, model_w, results):
    """
    Detection thread function to process frames
    
    Args:
        net: OpenCV DNN network
        cap: Video capture object
        model_h, model_w: Model dimensions
        results: Shared detection results object
    """
    while cap.isOpened():
        success, img0 = cap.read()
        if not success:
            break
 
        t1 = time.time()
        det_boxes, scores, ids = infer_image(net, img0, model_h, model_w, 
                                           thred_nms=0.4, thred_cond=0.4)
        t2 = time.time()
        fps_text = f"FPS: {1.0/(t2-t1):.2f}"
        
        results.update(det_boxes, scores, ids, fps_text)

if __name__=="__main__":
    # Class labels
    dic_labels = {0:'led',
                 1:'buzzer',
                 2:'teeth'}

    # Model parameters
    model_h = 640
    model_w = 640
    file_model = 'best-led-640.onnx'
    
    try:
        # Load the ONNX model
        net = cv2.dnn.readNet(file_model)
        print(f"Model {file_model} loaded successfully")
    except cv2.error as e:
        print(f"Error loading model: {e}")
        print(f"Make sure the model file '{file_model}' exists in the current directory")
        exit(1)
    
    # Initialize video capture
    video_source = 0  # Use camera index 0
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        exit(1)
        
    # Create shared detection results object
    detection_results = DetectionResults()
    
    # Start detection thread
    detector = Thread(target=detection_thread, 
                      args=(net, cap, model_h, model_w, detection_results),
                      daemon=True)
    detector.start()
    
    print("Detection started. Press 'q' to exit.")
    
    # Main loop for displaying results
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Get the latest detection results
        boxes, scores, ids, fps_text = detection_results.get()
        
        # Draw detection boxes
        for box, score, id in zip(boxes, scores, ids):
            if id in dic_labels:
                label = f'{dic_labels[id]}: {score:.2f}'
                plot_one_box(box, frame, color=(255, 0, 0), label=label, line_thickness=None)
        
        # Display FPS
        cv2.putText(frame, fps_text, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
        
        # Show the frame
        cv2.imshow("Object Detection", frame)
        
        # Check for exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")









