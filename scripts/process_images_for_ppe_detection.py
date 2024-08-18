import os
import cv2
import argparse
from ultralytics import YOLO

def parse_yolo_annotation(label_file):
    objects = []
    with open(label_file, 'r') as file:
        for line in file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            objects.append((class_id, x_center, y_center, width, height))
    return objects

def convert_yolo_bbox_to_coords(bbox, img_width, img_height):
    class_id, x_center, y_center, width, height = bbox
    xmin = int((x_center - width / 2) * img_width)
    ymin = int((y_center - height / 2) * img_height)
    xmax = int((x_center + width / 2) * img_width)
    ymax = int((y_center + height / 2) * img_height)
    return class_id, xmin, ymin, xmax, ymax

def save_cropped_image_and_label(image, objects, crop_coords, output_image_dir, output_label_dir, base_filename, index):
    xmin_crop, ymin_crop, xmax_crop, ymax_crop = crop_coords
    
    # Crop the image
    cropped_image = image[ymin_crop:ymax_crop, xmin_crop:xmax_crop]
    cropped_image_filename = f"{base_filename}_crop{index}.jpg"
    cv2.imwrite(os.path.join(output_image_dir, cropped_image_filename), cropped_image)
    
    # Update annotations for the cropped image
    updated_labels = []
    img_width = xmax_crop - xmin_crop
    img_height = ymax_crop - ymin_crop
    
    for obj in objects:
        class_id, xmin, ymin, xmax, ymax = convert_yolo_bbox_to_coords(obj, image.shape[1], image.shape[0])
        
        # Check if the object is within the crop
        if xmin < xmax_crop and ymin < ymax_crop and xmax > xmin_crop and ymax > ymin_crop:
            # Clip bbox to the cropped image boundaries
            xmin_clipped = max(xmin, xmin_crop) - xmin_crop
            ymin_clipped = max(ymin, ymin_crop) - ymin_crop
            xmax_clipped = min(xmax, xmax_crop) - xmin_crop
            ymax_clipped = min(ymax, ymax_crop) - ymin_crop
            
            # Convert to YOLO format (normalized coordinates)
            x_center = (xmin_clipped + xmax_clipped) / 2.0 / img_width
            y_center = (ymin_clipped + ymax_clipped) / 2.0 / img_height
            width = (xmax_clipped - xmin_clipped) / img_width
            height = (ymax_clipped - ymin_clipped) / img_height
            
            updated_labels.append(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Save the updated label file
    label_filename = f"{base_filename}_crop{index}.txt"
    with open(os.path.join(output_label_dir, label_filename), 'w') as f:
        for label in updated_labels:
            f.write(label + '\n')

def process_images(input_image_dir, input_label_dir, output_image_dir, output_label_dir, person_model):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    
    for filename in os.listdir(input_image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(input_image_dir, filename)
            label_path = os.path.join(input_label_dir, filename.replace('.jpg', '.txt'))
            
            # Load image and annotations
            image = cv2.imread(image_path)
            objects = parse_yolo_annotation(label_path)
            
            # Run person detection model to get person bounding boxes
            results = person_model(image)
            person_bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            
            # Save cropped images and corresponding labels
            for index, person_bbox in enumerate(person_bboxes):
                xmin, ymin, xmax, ymax = person_bbox
                crop_coords = (xmin, ymin, xmax, ymax)
                
                save_cropped_image_and_label(image, objects, crop_coords, output_image_dir, output_label_dir, filename.split('.')[0], index)

def main():
    parser = argparse.ArgumentParser(description="Crop person images and adjust labels accordingly.")
    parser.add_argument('--input_image_dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--input_label_dir', type=str, required=True, help='Directory containing input labels.')
    parser.add_argument('--output_image_dir', type=str, required=True, help='Directory to save cropped images.')
    parser.add_argument('--output_label_dir', type=str, required=True, help='Directory to save corresponding labels.')
    parser.add_argument('--person_model_path', type=str, required=True, help='Path to the person detection model.')
    parser.add_argument('--classes_file', type=str, required=True, help='Path to the classes.txt file.')

    args = parser.parse_args()

    # Load person detection model
    person_model = YOLO(args.person_model_path)

    # Load classes.txt and create class-to-index mapping
    with open(args.classes_file, 'r') as f:
        classes = f.read().strip().split()
    class_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}

    # Run the processing function
    process_images(args.input_image_dir, args.input_label_dir, args.output_image_dir, args.output_label_dir, person_model)

if __name__ == "__main__":
    main()