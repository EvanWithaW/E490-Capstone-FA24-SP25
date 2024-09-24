import os
import glob
import matplotlib.pyplot as plt
import cv2
import shutil


def move_files():
    txt_files = glob.glob(os.path.join(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\NO-LP\obj_train_data\no_lp", "*"))
    images_folder = r"D:\v2x-11-30-data\11-30-Parsed\no_lp"
    move_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\NO-LP\has-lp"
    # image_files = []
    num = 0
    for file_path in txt_files:
        root_path, file = os.path.split(file_path)
        name, ext = os.path.splitext(file)
        image_path = os.path.join(images_folder, name+".jpg")
        if not os.path.exists(image_path):
            move_path = os.path.join(move_folder, name+".txt")
            shutil.move(file_path, move_path)

def move_images():
    images_folder = r"D:\v2x-11-30-data\11-30-Parsed\no_lp"
    move_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\NO-LP\bad-images"
    txt_files = os.listdir(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\NO-LP\obj_train_data\no_lp")
    image_files = os.listdir(images_folder)
    num = 0

    for image in image_files:
        name, ext = os.path.splitext(image)
        if name+".txt" not in txt_files:
            shutil.move(os.path.join(images_folder, image), os.path.join(move_folder, image))
    print(num)

txt_files = os.listdir(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\NO-LP\obj_train_data\no_lp")
images = os.listdir(r"D:\v2x-11-30-data\11-30-Parsed\no_lp")
not_in = 0

for image in images:
    name, ext = os.path.splitext(image)
    if name+".txt" not in txt_files:
        not_in += 1
print(not_in)