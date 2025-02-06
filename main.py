import torch
import os
import cv2
import sys
import glob
import datetime
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import LPutility
import time as tm


# Path issue fix: https://stackoverflow.com/questions/57286486/i-cant-load-my-model-because-i-cant-put-a-posixpath
temp = ""
if sys.platform == "win32":
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

lp_weights = os.path.join("modelWeights", "LPbest.pt")
char_weights = os.path.join("modelWeights", "Charbest.pt")
# resnet_weights = os.path.join("weights", "resnet-classifier.pth")

# device = torch.device("mps")
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

lp_model = YOLO(lp_weights)
char_model = YOLO(char_weights)

### to run on the cpu
# resnet_classifier.load_state_dict(torch.load(resnet_weights, map_location=torch.device('cpu')))
# lp_model.eval()
# char_model.eval()`
lp_model.to(device)
char_model.to(device)

image_size = 32
transforms = A.Compose([
        # A.Affine(shear={"x":15}, p=1.0),
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


# image_paths = glob.glob(os.path.join(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\test\images", "*"))
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
        loadingImageTimes.append(end-start)
        # print("Time spent loading image",end-start)

        start = tm.time()
        # lp_pred = lp_model(image)
        lp_pred = lp_model(image,verbose=False)
        end = tm.time()
        predictingLPTimes.append(end-start)

        # print("Time spent predicting LP",end-start)



    # lp_pred[0].show()
        start = tm.time()
        lp_crop = LPutility.get_crop_lp(lp_pred, image)
        end = tm.time()
        croppingLPTimes.append(end-start)
        # print("Time spent cropping LP",end-start)

        # debugging
        # cv2.imshow("Cropped Image", lp_crop)
        # cv2.waitKey(0)

        ## so if this doesnt pass through then there was no license plate found in that photo.
        if lp_crop.size > 0:
            # invert the image
            # inverted = ~lp_crop
            # histogram equalize the image
            # he = LPutility.HE(lp_crop)
            # invert the histogram equalized image
            # inv_he = LPutility.HE(inverted)
            # run character detection on the inverted histogram equalized image This had the best results for detecting characters on plates.

            start = tm.time()
            # char_pred = char_model(lp_crop)
            char_pred = char_model(lp_crop,verbose=False)

            pred, conf = LPutility.directPredict(char_pred)
            conf = (conf.float()*1000).int().item()
            end = tm.time()
            predictingCharTimes.append(end-start)
            # print("Time spent predicting characters",end-start)
            # print("----")
            # print(pred)
            # cv2.imshow("Cropped Image", lp_crop)
            # cv2.waitKey(0)
            # time.wait(500)
            # pred = utils.predict_chars(char_crops, transforms=transforms, device=device)
            # print(f"CONF: {conf}")
            file.write(f"{pred},{name},{conf}\n")
            count += 1
            endglobal = tm.time()
            globalrunTimes.append(endglobal-startglobal)
            if count % 1000 == 0:
                # end = time.time()
                # print("Time spent avg",end-start/count)
                # start = time.time()
                print(f"Finished prediction {count}")
                print(f"""
                Average time loading image: {sum(loadingImageTimes)/len(loadingImageTimes)}
                Average time predicting LP: {sum(predictingLPTimes)/len(predictingLPTimes)}
                Average time cropping LP: {sum(croppingLPTimes)/len(croppingLPTimes)}
                Average time predicting Char: {sum(predictingCharTimes)/len(predictingCharTimes)}
                Average time running everything: {sum(globalrunTimes)/len(globalrunTimes)}
                """)
print(f"Results written to model-runs/{filename}")        

if sys.platform == "win32":
    pathlib.PosixPath = temp
        