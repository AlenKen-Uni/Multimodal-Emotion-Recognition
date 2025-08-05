import os
import cv2
import numpy as np

from ultralytics import YOLO

# load the pre-trained DNN model for face detection
dir = os.path.dirname(os.path.abspath(__file__))
net = cv2.dnn.readNetFromCaffe(
    os.path.join(dir.split("MER")[0] + "MER", "deploy.prototxt"),
    os.path.join(dir.split("MER")[0] + "MER", "res10_300x300_ssd_iter_140000.caffemodel")
)
# load the Haar Cascade classifier for face detection
face_haarcascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
# load the YOLOv8 face detection model
yolo = YOLO(os.path.join(dir.split("MER")[0] + "MER", "yolov8n-face.pt"))

def resize_square_img(img:np.ndarray, size:int=224)-> np.ndarray:
    """ Resize image to a square of given size, padding if necessary """
    h, w = img.shape[:2]
    sz = max(h, w)
    dw, dh = sz - w, sz - h
    top, bottom = dh//2, dh - dh//2
    left, right = dw//2, dw - dw//2

    img_square = cv2.copyMakeBorder(
        img,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0] if len(img.shape) == 3 else [0]
    )
    
    # select interpolation method based on image size
    # using cubic for smaller images and area for larger ones
    interp = cv2.INTER_CUBIC if max(h, w) < size else cv2.INTER_AREA
    img_resized = cv2.resize(img_square, (size, size), interpolation=interp)
    
    return img_resized
    
def detect_face_using_dnn(frame:np.ndarray, conf_thresh=0.5) -> np.ndarray:
    """ Detect faces in a frame using OpenCV's DNN module with a pre-trained Res10 SSD model"""
    face_list = []
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    net.setInput(blob)
    detections = net.forward()[0, 0]
        
    for det in detections:
        score = float(det[2])
        if score < conf_thresh:
            continue
        # extract box in pixel coords
        x1 = max(0, int(det[3] * w))
        y1 = max(0, int(det[4] * h))
        x2 = min(w, int(det[5] * w))
        y2 = min(h, int(det[6] * h))
            
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # resize to square via padding
        face_resized = resize_square_img(face_crop, size=224)
        face_list.append(face_resized)
    
    return np.array(face_list, dtype=np.uint8)

def detect_face_using_haarcascade(frame:np.ndarray) -> np.ndarray:
    """ Detect faces in a frame using OpenCV's Haar Cascade classifier """
    face_list = []

    # detect faces using Face Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haarcascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        face_crop = frame[y:y+h, x:x+w] # (h, w, 3)
        
        # resize to square via padding
        face_resized = resize_square_img(face_crop, size=224)
        face_list.append(face_resized)
        
    return np.array(face_list, dtype=np.uint8)

def detect_face_using_yolo(frame:np.ndarray, conf_thresh:float=0.25) -> np.ndarray:
    face_list = []
    # single-frame inference
    results = yolo.predict(
        source=frame,
        imgsz=480,         # small image → fast backbone convs
        device="cuda:0",   # run on GPU
        half=True,         # FP16 mode
        max_det=10,        # only keep top 10 faces per frame
        classes=[0],       # 0 is the “face” class in yolov8n-face.pt
        # stream=True,     # keep model weights in GPU memory
        verbose=False,
        augment=False      # turn off TTA
    )[0]

    # filter by confidence and extract boxes
    for box in results.boxes.data:  # tensor of shape (num,6): [x1,y1,x2,y2,conf,class]
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        if conf <= conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue
        
        # resize to square via padding
        face_resized = resize_square_img(face_crop, size=224)
        face_list.append(face_resized)
        
    return np.array(face_list, dtype=np.uint8)