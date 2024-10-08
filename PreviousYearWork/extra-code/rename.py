import os
import glob

file_list = os.listdir("roboflow_image_dataset")
dir = "roboflow_image_dataset"
num = 0

for file in file_list:
    os.rename(f"{dir}/{file}", f"{dir}/{num}.jpg")
    num += 1

