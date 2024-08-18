import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir, test_size=0.2, valid_size=0.2):
    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')

    # Create output directories
    os.makedirs(os.path.join(output_dir, 'train/images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'train/labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid/images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid/labels'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test/images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test/labels'), exist_ok=True)

    # Get list of image files
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    image_files.sort()

    # Split the dataset
    train_files, temp_files = train_test_split(image_files, test_size=test_size, random_state=42)
    valid_files, test_files = train_test_split(temp_files, test_size=test_size / (test_size + valid_size), random_state=42)

    def copy_files(file_list, destination_folder):
        for file in file_list:
            # Copy images
            shutil.copy(os.path.join(images_dir, file), os.path.join(destination_folder, 'images', file))
            # Copy corresponding annotations
            label_file = os.path.splitext(file)[0] + '.txt'
            label_file_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_file_path):
                shutil.copy(label_file_path, os.path.join(destination_folder, 'labels', label_file))
            else:
                print(f"Warning: Label file {label_file} not found for {file}")

    # Copy files to respective folders
    copy_files(train_files, os.path.join(output_dir, 'train'))
    copy_files(valid_files, os.path.join(output_dir, 'valid'))
    copy_files(test_files, os.path.join(output_dir, 'test'))

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