from ultralytics import YOLO
import cv2
import os


model = YOLO("yolov8n.pt")
images_folder = ".\input-data"

for image_file in os.listdir(images_folder):
    if image_file.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image extensions
        image_path = os.path.join(images_folder, image_file)

        results = model(image_path, show=True)
        cv2.waitKey(0)
