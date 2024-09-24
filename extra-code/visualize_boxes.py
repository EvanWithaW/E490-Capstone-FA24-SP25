import os
import glob
import cv2
import matplotlib.pyplot as plt
import utility
import sys
import shutil


def on_press(event,figure, name):
        # print('press', event.key)
        sys.stdout.flush()
        image = name + ".jpg"
        txt = name + ".txt"
        images = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\images"
        labels = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\labels"
        not_in = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\not-in"

        if event.key == 'x':
            plt.close(figure)
        # elif event.key == "w":
        #      print(name)
        # elif event.key == "c":
        #      shutil.move(os.path.join(not_in, image), os.path.join(images, image))
        #      shutil.move(os.path.join(not_in, txt), os.path.join(labels, txt))

ann_folder = r"D:\v2x-11-30-data\TEST-LABELS\CHAR-DETECT"
images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\test\images"
ann_files = glob.glob(os.path.join(ann_folder, "*.txt"))

for ann_file in ann_files:
    with open(ann_file, "r") as file:
        base, txt_file = os.path.split(ann_file)
        name, ext = os.path.splitext(txt_file)
        image_path = os.path.join(images_folder, name+".png")
        if not os.path.exists(image_path):
             continue
        image = cv2.imread(image_path)
        for line in file.readlines():
            xmin, ymin, xmax, ymax = utility.annotation_to_points(image, line)
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        
        figure = plt.figure(figsize=(8, 8))
        figure.canvas.mpl_connect("key_press_event", lambda event: on_press(event, figure, name))
        plt.imshow(image)
        plt.show()