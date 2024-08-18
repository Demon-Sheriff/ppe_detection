import os
import xml.etree.ElementTree as ET
import argparse

def convert_annotation(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(input_dir, filename))
        root = tree.getroot()

        output_filename = os.path.join(output_dir, filename.replace('.xml', '.txt'))
        with open(output_filename, 'w') as out_file:
            for obj in root.iter('object'):
                cls = obj.find('name').text
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                     float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
                # Example conversion function; modify as per your classes file
                yolo_box = convert_to_yolo_format(b, root.find('size'))
                out_file.write(f"{cls_id} " + " ".join([str(a) for a in yolo_box]) + '\n')

def convert_to_yolo_format(bbox, size):
    dw = 1.0 / int(size.find('width').text)
    dh = 1.0 / int(size.find('height').text)
    x = (bbox[0] + bbox[2]) / 2.0 * dw
    y = (bbox[1] + bbox[3]) / 2.0 * dh
    w = (bbox[2] - bbox[0]) * dw
    h = (bbox[3] - bbox[1]) * dh
    return x, y, w, h

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC to YOLOv8 format.")
    parser.add_argument('input_dir', type=str, help="Directory with PascalVOC annotation XML files")
    parser.add_argument('output_dir', type=str, help="Directory to save YOLOv8 annotation files")
    args = parser.parse_args()

    convert_annotation(args.input_dir, args.output_dir)
