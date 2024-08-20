# PPE Detection with YOLOv8

### Overview

This project aims to detect Personal Protective Equipment (PPE) in images using the YOLOv8 model. The detection process involves two main steps:

1. Detecting persons in images using a pre-trained person detection model.
2. Detecting PPE within the cropped regions of detected persons.

## Project Structure
```
PPE-Detection/
│
├── data/
│   ├── images/                  # Directory containing original images
│   ├── labels/                  # Directory containing original YOLO annotations
│   ├── train/                   # Training data after splitting
│   ├── valid/                   # Validation data after splitting
│   ├── test/                    # Test data after splitting
│   └── data.yaml                # Dataset configuration file for YOLOv8
│
├── models/
│   ├── person_model.pt          # Pre-trained person detection model
│   └── ppe_model.pt             # Pre-trained PPE detection model (after training)
│
├── scripts/
│   ├── process_images.py        # Script to process images, detect persons, and adjust PPE labels
│   ├── split_dataset.py         # Script to split the dataset into training, validation, and test sets
│   ├── inference.py             # Script to run inference on images and detect PPE
│   └── train_model.sh           # Shell script to train the YOLOv8 model for PPE detection
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── results/                     # Directory to store inference results
```
## Setup

#### Prerequisite
- Python 3.7+
- CUDA-compatible GPU (optional but recommended)
- Python packages listed in `requirements.txt`

#### Installation
