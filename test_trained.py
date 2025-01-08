from ultralytics import YOLO
import cv2

model_name = "car_model_75ep_640x640_gpu.pt"

model = YOLO(model_name)
model.to('cpu')


# Path to your test image
image_path = 'test/all_cars_3.jpg'

# Save the output image with predictions
results = model.predict(source=image_path, save=True,
                        conf=0.40, iou=0.4)
