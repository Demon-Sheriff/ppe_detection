import cv2
import os
import argparse
from ultralytics import YOLO

# List of class names for the PPE model
ppe_class_names = ['hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 
                   'ppe-suit', 'ear-protector', 'safety-harness']

def inference(input_dir, output_dir, person_model_path, ppe_model_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load YOLO models for person detection and PPE detection
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)

    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)
        image = cv2.imread(image_path)

        # Step 1: Detect persons in the full image
        person_results = person_model(image)

        for i, person_bbox in enumerate(person_results[0].boxes):
            xmin, ymin, xmax, ymax = person_bbox.xyxy[0].tolist()
            confidence = person_bbox.conf[0]  # Person detection confidence

            # Draw bounding box and confidence score for person
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 1)
            cv2.putText(image, f'Person: {confidence:.2f}', 
                        (int(xmin), int(ymin) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 0, 0), 
                        1)

            # Step 2: Crop the detected person region
            cropped_img = image[int(ymin):int(ymax), int(xmin):int(xmax)]

            # Step 3: Detect PPE on the cropped person region
            ppe_results = ppe_model(cropped_img)

            # Convert PPE bounding boxes back to full image coordinates
            for ppe_bbox in ppe_results[0].boxes:
                pxmin, pymin, pxmax, pymax = ppe_bbox.xyxy[0].tolist()
                ppe_conf = ppe_bbox.conf[0]  # Confidence score for PPE detection
                ppe_class_id = int(ppe_bbox.cls[0])  # Class ID for PPE detection
                ppe_label = ppe_class_names[ppe_class_id]  # Class name

                # Draw bounding box and confidence score for PPE on the full image
                cv2.rectangle(image, 
                              (int(xmin + pxmin), int(ymin + pymin)), 
                              (int(xmin + pxmax), int(ymin + pymax)), 
                              (0, 255, 0), 2)
                cv2.putText(image, f'{ppe_label}: {ppe_conf:.2f}', 
                            (int(xmin + pxmin), int(ymin + pymin) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.4, 
                            (0, 255, 0), 
                            1)

        # Save the final image with detections to the output directory
        output_image_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_image_path, image)

def main():
    parser = argparse.ArgumentParser(description="Perform person and PPE detection on input images.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output images with detection results.')
    parser.add_argument('--person_model_path', type=str, required=True, help='Path to the person detection model.')
    parser.add_argument('--ppe_model_path', type=str, required=True, help='Path to the PPE detection model.')

    args = parser.parse_args()

    # Run inference
    inference(args.input_dir, args.output_dir, args.person_model_path, args.ppe_model_path)

if __name__ == "__main__":
    main()