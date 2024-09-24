import torch
import glob
import os
import cv2
import math
import sys
from scripts.utility import get_bounding_box_data

"""
    Use YOLOv5 to create annotations for CVAT annotation tool
    arg1: YOLOv5 model weights file, example filename: best.pt

    yolo annotation file format: class_number x_center y_center box_width box_height
"""

if len(sys.argv) < 2:
    print("Error. Need to provide YOLOv5 model weights file.")
    sys.exit()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

weights = sys.argv[1]
model = torch.hub.load('ultralytics/yolov5', 'custom', weights)
model.eval()    
model.to(device)

image_files = glob.glob(os.path.join(r"D:\v2x-11-30-data\11-30-Parsed\different-length", '*.*'))
image_annotation_file = open(r"D:\v2x-11-30-data\11-30-Parsed\different-length-ann\cvat-annotations\train.txt", "w")
batch_size = 16
batches = [image_files[i:i+batch_size] for i in range(0, len(image_files), batch_size)]
# print(f"Number of Batches: {len(batches)}")

# batches = batches[5:10]
batch_num = 0
for batch in batches:
    results = model(batch)
    batch_num += 1

    if batch_num % 50 == 0:
        print(batch_num)

    for pred, im in zip(results.xyxy, batch):
        if pred.numel() == 0:
            continue
        
        img = cv2.imread(im)
        pred = pred.tolist()  
        boxes = get_bounding_box_data(pred, img, 0, model="char")

        ### CVAT Annotation folder format: https://opencv.github.io/cvat/docs/manual/advanced/formats/format-yolo/
        directory = r"D:\v2x-11-30-data\11-30-Parsed\different-length-ann\cvat-annotations\obj_train_data/"
        filename = im.split("\\")[-1]
        filename = filename.split(".")[0]   # remove image type ending

        # write image location for CVAT
        image_annotation_file.write(f"obj_train_data/{filename}.png\n")

        with open(f"{directory}/{filename}.txt", "w") as file:
            image = cv2.imread(im)
            image_height, image_width, channels = image.shape

            for box in boxes:
                bounding_box = box[0]
                confidence = box[1]
                class_number = int(box[2])
                
                xmin, ymin, xmax, ymax = bounding_box
                box_width = xmax - xmin
                box_height = ymax - ymin
                x_center = xmin + (box_width/2)
                y_center = ymin + (box_height/2)

                # normalize values
                box_width = box_width / image_width
                box_height = box_height / image_height
                x_center = x_center / image_width
                y_center = y_center / image_height
                
                file.write(f"{class_number} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
# model.to("cpu")
image_annotation_file.close()


