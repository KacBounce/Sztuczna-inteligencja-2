import os
import random
import shutil
from pathlib import Path

# Set paths
train_folder = 'datasets/images/train'
val_folder = 'datasets/images/val'
# Minimum number of images required per class in the train set
min_images_per_class = 10
validation_percentage = 0.2  # Percentage of images to move to the validation set

# Ensure the validation folder exists
os.makedirs(val_folder, exist_ok=True)


def move_images_to_val(class_folder, images_to_move):
    """Move selected images to the validation folder."""
    for img_file in images_to_move:
        # Move the image to the validation folder
        shutil.move(str(img_file), os.path.join(
            val_folder, class_folder.name, img_file.name))
        # Move corresponding label file as well
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            shutil.move(str(label_file), os.path.join(
                val_folder, class_folder.name, label_file.name))
        print(f"Moved {img_file} and its label to {
              val_folder}/{class_folder.name}")


def remove_class_with_insufficient_images(class_folder):
    """Remove a class folder with insufficient images from the train folder."""
    print(f"Removing class {class_folder.name} due to insufficient images.")
    shutil.rmtree(class_folder)


def process_train_directory():
    """Process the train directory: move images to val and remove classes with insufficient images."""
    for class_folder in Path(train_folder).iterdir():
        if class_folder.is_dir():
            images = list(class_folder.glob('*.jpg'))
            num_images = len(images)
            print(f"Class {class_folder.name}: {num_images} images found.")

            # Eliminate classes with fewer images than the threshold
            if num_images < min_images_per_class:
                remove_class_with_insufficient_images(class_folder)
                continue

            # Calculate how many images to move to the val set
            num_to_move = int(num_images * validation_percentage)

            # Select random images to move to the validation set
            images_to_move = random.sample(images, num_to_move)

            # Ensure the class folder exists in the validation set
            os.makedirs(os.path.join(
                val_folder, class_folder.name), exist_ok=True)

            # Move the selected images to the validation folder
            move_images_to_val(class_folder, images_to_move)


# Run the process
process_train_directory()
