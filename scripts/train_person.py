from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' for a smaller model; choose based on your need

# Train the model using the person_detection.yaml configuration
model.train(data="C:\\Users\\Anant Shukla\\Desktop\\Object_Detection\\config\\person.yaml", epochs=50, imgsz=640)
# Save the model weights
model.save("./weights/person_detection.pt")
