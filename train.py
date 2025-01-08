# Import the YOLO class from the ultralytics library
from ultralytics import YOLO

# Define paths and parameters
dataset_yaml = "dataset.yaml"  # Path to your dataset YAML file
# Pre-trained model to use (e.g., yolov8n, yolov8m, yolov8l)
model_name = "yolov8n.pt"
epochs = 75                    # Number of epochs
img_size = 640                 # Image size for training

def main():
    # Load the YOLO model
    model = YOLO(model_name)
    model.to('cuda')

    # Train the model
    model.train(data=dataset_yaml, epochs=epochs, imgsz=img_size)

    # Evaluate the model on the validation set
    metrics = model.val()
    model.save(f"car_model_{epochs}ep_{img_size}x{img_size}_gpu.pt")
    # Perform inference on a test image
    # Replace with the path to your test image
    test_image = "test\ad.jpg"
    # Save the output image with predictions
    results = model.predict(source=test_image, save=True)
    
if __name__ == '__main__':
    # This is the entry point for the script
    from multiprocessing import freeze_support
    freeze_support()  # Necessary for Windows if freezing the script into an executable
    main()  # Call the main function that contains the training logic


