import datetime
import glob
import os
import sys
import time as tm
import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from ultralytics import YOLO
import LPutility

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

lp_weights = os.path.join("modelWeights", "LPbest.pt")
char_weights = os.path.join("modelWeights", "Charbest.pt")

lp_model = YOLO(lp_weights)
char_model = YOLO(char_weights)

image_size = 32
transforms = A.Compose([
    A.LongestMaxSize(max_size=image_size),
    A.PadIfNeeded(
        min_height=image_size,
        min_width=image_size,
        border_mode=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    ),
    A.Normalize(
        mean=[0, 0, 0],
        std=[1, 1, 1],
        max_pixel_value=255,
    ),
    ToTensorV2()
])

image_paths = []
if len(sys.argv) < 2:
    print("Missing Argument: image or images folder path")

path = sys.argv[1]
if os.path.isfile(path):
    image_paths = [path]
elif os.path.isdir(path):
    image_paths = glob.glob(os.path.join(path, "*"))

if not os.path.isdir("model-runs"):
    os.mkdir("model-runs")

timestamp = str(datetime.datetime.now())
date, time = timestamp.split(".")[0].split(" ")
time = time.replace(":", "-")
filename = f"Run---{date}---{time}.csv"
count = 0
loadingImageTimes = []
predictingLPTimes = []
croppingLPTimes = []
predictingCharTimes = []
globalrunTimes = []

with open(os.path.join("model-runs", filename), "w") as file:
    file.write("PRED,IMAGE,CONF\n")
    for image_path in image_paths:
        startglobal = tm.time()
        start = tm.time()
        base, image_name = os.path.split(image_path)
        name, ext = os.path.splitext(image_name)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        end = tm.time()
        loadingImageTimes.append(end - start)

        start = tm.time()
        lp_pred = lp_model.predict(image, device=device, verbose=False)
        end = tm.time()
        predictingLPTimes.append(end - start)

        start = tm.time()
        lp_crop = LPutility.get_crop_lp(lp_pred, image)
        end = tm.time()
        croppingLPTimes.append(end - start)

        ## so if this doesnt pass through then there was no license plate found in that photo.
        if lp_crop.size > 0:
            start = tm.time()
            char_pred = char_model.predict(lp_crop, device=device, verbose=False)

            pred, conf = LPutility.directPredict(char_pred)
            conf = (conf.float() * 1000).int().item()
            end = tm.time()
            predictingCharTimes.append(end - start)
            file.write(f"{pred},{name},{conf}\n")
            count += 1
            endglobal = tm.time()
            globalrunTimes.append(endglobal - startglobal)
            if count % 10000 == 0:
                print(f"Finished prediction {count}")
                print(f"""
                Average time loading image: {sum(loadingImageTimes) / len(loadingImageTimes)}
                Average time predicting LP: {sum(predictingLPTimes) / len(predictingLPTimes)}
                Average time cropping LP: {sum(croppingLPTimes) / len(croppingLPTimes)}
                Average time predicting Char: {sum(predictingCharTimes) / len(predictingCharTimes)}
                Average time running everything: {sum(globalrunTimes) / len(globalrunTimes)}
                """)

print(f"Results written to model-runs/{filename}")
print(f"Finished prediction {count}")
print(f"""
Average time loading image: {sum(loadingImageTimes) / len(loadingImageTimes)}
Average time predicting LP: {sum(predictingLPTimes) / len(predictingLPTimes)}
Average time cropping LP: {sum(croppingLPTimes) / len(croppingLPTimes)}
Average time predicting Char: {sum(predictingCharTimes) / len(predictingCharTimes)}
Average time running everything: {sum(globalrunTimes) / len(globalrunTimes)}
""")