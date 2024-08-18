from ultralytics import YOLO
import os

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for a smaller model; choose based on your need

# Train the model using the person_detection.yaml configuration
model.train(data='/content/drive/MyDrive/configs/person.yaml', epochs=50, imgsz=640)

# Ensure the directory exists for saving the model weights
weights_dir = '/content/drive/MyDrive/weights'
os.makedirs(weights_dir, exist_ok=True)

# Save the model weights
model.save(os.path.join(weights_dir, 'person_detection.pt'))
