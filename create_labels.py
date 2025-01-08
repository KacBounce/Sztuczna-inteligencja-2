import os
from pathlib import Path
import torch
from PIL import Image

# Load YOLOv5 pre-trained model (you can choose other versions like yolov8 if necessary)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set paths
# Path to the folders containing car image folders (images/train, images/val, images/test)
image_folders = ['datasets/images/train', 'datasets/images/val']
# Path to the folder where labels will be saved
output_folder = 'datasets/labels'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to convert bounding boxes to YOLO format (normalized x_center, y_center, width, height)


def convert_yolo_format(box, img_width, img_height):
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]

    # Normalize by image dimensions
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height

    return x_center, y_center, width, height

# Function to process images in each folder and generate labels


def process_images():
    # Iterate through each image folder (train, val)
    for image_folder in image_folders:
        print(f"Processing folder: {image_folder}")

        # Create corresponding subfolder in output folder for labels (train or val)
        label_folder = os.path.join(output_folder, Path(image_folder).name)
        os.makedirs(label_folder, exist_ok=True)

        # Iterate through all car class folders in the current image folder
        for idx, class_folder in enumerate(Path(image_folder).iterdir()):
            if class_folder.is_dir():  # Check if it's a directory (folder)
                # Get class ID based on folder order (first folder = class 0, second = class 1, etc.)
                class_id = idx

                print(f"Processing class folder: {
                      class_folder.name}, Assigned Class ID: {class_id}")

                # Ensure the same subfolder structure exists in the output folder
                label_subfolder = os.path.join(label_folder, class_folder.name)
                os.makedirs(label_subfolder, exist_ok=True)

                # Iterate through all image files in the folder
                for image_file in class_folder.glob('*.jpg'):
                    img = str(image_file)

                    # Perform inference on the image using YOLO
                    results = model(img)

                    # Get image dimensions
                    img_width, img_height = results.ims[0].shape[1], results.ims[0].shape[0]

                    # Create the label file path (YOLO format, same name as image file but with .txt extension)
                    label_file = os.path.join(
                        label_subfolder, f"{image_file.stem}.txt")

                    # Open the label file for writing
                    with open(label_file, 'w') as f:
                        # Iterate through detected boxes and save in YOLO format
                        for *box, conf, cls in results.xyxy[0].tolist():
                            # Convert box to YOLO format
                            x_center, y_center, width, height = convert_yolo_format(
                                box, img_width, img_height)

                            # Write the label with class_id and normalized coordinates
                            f.write(f"{class_id} {x_center} {
                                    y_center} {width} {height}\n")

                    print(f"Processed {image_file} -> {label_file}")


# Run the function to process images
process_images()
