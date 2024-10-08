import os
import glob
import json
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import sys


def find_char_dist():
    root_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR-CLASSIFY\characters"
    folders = os.listdir(root_folder)
    char_dist = {}
    for folder in folders:
        char_dist[folder] = len(os.listdir(os.path.join(root_folder, folder)))

    with open("train-char-dist.json", "w") as file:
        json.dump(char_dist, file)

def get_size(root_folder):
    size = 0
    for ele in os.scandir(root_folder):
        size += os.path.getsize(ele)
    return size


transforms = A.Compose([
    A.ColorJitter(p=0.5),
    A.Affine(shear=(0.3,0.4), p=0.5),
    A.RandomScale((0.7, 0.9)),
    A.Rotate(limit=(-15, 15), p=0.3),
    A.Sharpen(p=0.5),
    # A.LongestMaxSize(max_size=image_size),
    # A.PadIfNeeded(
    #     min_height=image_size,
    #     min_width=image_size,
    #     border_mode=cv2.BORDER_CONSTANT,
    #     value=(0, 0, 0),
    # ),
    # A.Normalize(
    #     mean=[0, 0, 0],
    #     std=[1, 1, 1],
    #     max_pixel_value=255,
    # ),
])

root_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR-CLASSIFY\characters"
char_dist = ""
with open("train-char-dist.json", "r") as file:
    char_dist = json.load(file)

# key = sys.argv[1]
characters = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
maximum = 100000
for key in characters:
    images = os.listdir(os.path.join(root_folder, key))
    start = len(images)
    print()
    print(key)
    print("--------")

    while start < maximum:
        for image in images:
            if start > maximum:
                break
                
            if start % 1000 == 0:
                print(start)

            name = image[0]
            img = cv2.imread(os.path.join(root_folder, key, image))
            augmented = transforms(image=img)["image"]
            save_name = name + "-aug-"+str(start)+".png"
            save_path = os.path.join(root_folder, key, save_name)
            cv2.imwrite(save_path, augmented)
            start += 1


