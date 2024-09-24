# YOLO Setup: https://medium.com/analytics-vidhya/training-a-custom-object-detection-model-with-yolo-v5-aa9974c07088
# YOLO Custom Training: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#local-logging
# YOLO Object Detection Tutorial: https://docs.ultralytics.com/datasets/detect/
# Inference using custom trained model


import torch
import glob
import os

model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5/runs/train/exp4/weights/best.pt')

# load trained weights
# model.load_state_dict(torch.load('yolov5/runs/train/exp4/weights/best.pt')['model'].state_dict())

# set for inference
model.eval()    

image_files = glob.glob(os.path.join('images', '*.*'))
images = []
for file in image_files:
    images.append(file.split("\\")[1])                  
print(images)

# img = "images/LP.jpg"

for img in image_files:
    print(f"Image: {img}")
    results = model(img)
    results.show()
    print(results.pandas().xyxy[0])

