import os
import random
import shutil
import argparse

def split_dataset(image_dir, label_dir, output_dir, train_ratio=0.8):
    # Create output directories
    train_images_dir = os.path.join(output_dir, 'train/images')
    val_images_dir = os.path.join(output_dir, 'val/images')
    train_labels_dir = os.path.join(output_dir, 'train/labels')
    val_labels_dir = os.path.join(output_dir, 'val/labels')

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Get list of all images
    images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # Shuffle the images to ensure random split
    random.shuffle(images)

    # Split the data
    train_size = int(len(images) * train_ratio)
    train_images = images[:train_size]
    val_images = images[train_size:]

    def copy_files(image_list, images_dir, labels_dir, output_images_dir, output_labels_dir):
        for image in image_list:
            image_path = os.path.join(images_dir, image)
            label_path = os.path.join(labels_dir, image.replace('.jpg', '.txt'))
            
            # Check if image exists
            if not os.path.isfile(image_path):
                print(f"Warning: Image file {image_path} does not exist.")
                continue
            
            # Copy image
            shutil.copy(image_path, os.path.join(output_images_dir, image))
            
            # Check if label file exists
            if os.path.isfile(label_path):
                # Copy label
                shutil.copy(label_path, os.path.join(output_labels_dir, image.replace('.jpg', '.txt')))
            else:
                print(f"Warning: Label file {label_path} does not exist.")

    # Copy files to the respective directories
    copy_files(train_images, image_dir, label_dir, train_images_dir, train_labels_dir)
    copy_files(val_images, image_dir, label_dir, val_images_dir, val_labels_dir)

    print(f"Training set: {len(train_images)} images")
    print(f"Validation set: {len(val_images)} images")

# Example usage
# split_dataset('/content/drive/MyDrive/datasets/images', '/content/drive/MyDrive/datasets/yolo_labels', '/content/drive/MyDrive/data')


def main():
    parser = argparse.ArgumentParser(description="Split dataset into train, valid, and test sets.")
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing "images" and "labels" subdirectories.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for train, valid, and test splits.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')
    parser.add_argument('--valid_size', type=float, default=0.2, help='Proportion of the dataset to include in the validation split.')

    args = parser.parse_args()

    # Run the dataset split
    split_dataset(args.input_dir, args.output_dir, test_size=args.test_size, valid_size=args.valid_size)

if __name__ == "__main__":
    main()