import torch
import glob
import os
import cv2
import sys
from scripts.utility import get_bounding_box_data


if len(sys.argv) < 3:
    print("Missing Image Folder and/or Label Destination Folder")
    sys.exit()


import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

weights = r"C:\Users\Jed\Desktop\capstone_project\CapstoneLPR\best_weights\char-detect-M-20k.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'custom', weights)#, force_reload=True)
model.eval()    
model.to(device)

image_files = glob.glob(os.path.join(sys.argv[1], '*.*'))
num_files = len(image_files)
dest_folder = sys.argv[2]
for index, image_path in enumerate(image_files):
    if index % 100 == 0:
        print(f"{index}/{num_files}")
    results = model(image_path)

    base, file = os.path.split(image_path)
    name, ext = os.path.splitext(file)

    for pred in results.xyxy:
        if pred.numel() == 0:
            continue
        
        image = cv2.imread(image_path)
        pred = pred.tolist()  
        boxes = get_bounding_box_data(pred, image, 0, model="char")
        image_height, image_width, channels = image.shape

        with open(os.path.join(dest_folder, name+".txt"), "w") as f:
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

                f.write(f"{class_number} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")