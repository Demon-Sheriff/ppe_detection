import xml.etree.ElementTree as ET
import os
import argparse

# def print_tree(element, level=0):
#     print("  " * level + element.tag)
#     for child in element:
#         print_tree(child, level + 1)

def get_size_props(root):
    size = root.find('size')
    if size is not None:
        w = size.find('width').text
        h = size.find('height').text
        d = size.find('depth').text
    else:
        print("Size information in file not found")
        return None
    return int(w), int(h), int(d)

def convert_bbox(x1, y1, x2, y2, w, h):
    return [(x2 + x1)/(2*w), (y2 + y1)/(2*h), (x2 - x1)/w, (y2 - y1)/h]

def get_class_props(root, output_file):
    classes = ['person', 'hard-hat', 'gloves', 'mask', 'glasses', 'boots', 'vest', 'ppe-suit', 'ear-protector', 'safety-harness']
    w, h, _ = get_size_props(root)
    
    with open(output_file, 'w') as f:
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name in classes:
                class_id = classes.index(name)
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # Convert to YOLO format
                bbox = convert_bbox(xmin, ymin, xmax, ymax, w, h)
                
                # Write to file
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.xml'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename[:-4] + '.txt')

            tree = ET.parse(input_path)
            root = tree.getroot()

            get_class_props(root, output_path)
            print(f"Converted {filename} to {os.path.basename(output_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pascal VOC XML to YOLO txt format")
    parser.add_argument("input_dir", help="Input directory containing Pascal VOC XML files")
    parser.add_argument("output_dir", help="Output directory for YOLO txt files")
    args = parser.parse_args()

    process_directory(args.input_dir, args.output_dir)
    print("Conversion complete.")