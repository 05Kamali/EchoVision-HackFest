import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for more accuracy

# Function to announce detected objects using espeak-ng
def ann(object_name):
    os.system(f'espeak-ng "{object_name}"')

# Load class names
classfile ="D:\Set\Titanic\model\model\coco.names"
with open(classfile, 'r') as f:
    classname = f.read().strip().split("\n")

# Initialize video capture
cap = cv2.VideoCapture(1)  # Use 0 for default USB camera
frame_skip = 15  # Skip frames for efficiency
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    frame_count += 1
    if frame_count % frame_skip != 10:
        continue  # Skip frames for better performance

    # Run YOLOv8 detection
    results = model.predict(frame, conf=0.3)  # Confidence threshold = 0.3

    detected_objects = set()
    for result in results:
        for box in result.boxes:
            classId = int(box.cls[0])
            object_name = classname[classId].upper()
            
            if object_name not in detected_objects:
                detected_objects.add(object_name)
                ann(object_name)  # Announce detected object

    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
