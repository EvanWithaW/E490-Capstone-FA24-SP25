import os
import cv2
import matplotlib.pyplot as plt
import utility as utils
import json


def make_dirs(characters, path):
    for char in characters:
        os.mkdir(os.path.join(path, char))

def vertical_sort(boxes):
    num_boxes = len(boxes)

    for i in range(0, num_boxes - 1):
        box1 = boxes[i]
        box2 = boxes[i + 1]
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        box1_height = ymax1 - ymin1
        box2_height = ymax2 - ymin2

        # 0.25 to account for the possibility of vertical overlap
        if (ymin1 >= (ymax2 - 0.25*box2_height)):
            # print("ran")
            if xmax1 >= xmin2 and xmax1 <= xmax2:
                temp = boxes[i]
                boxes[i] = boxes[i + 1]
                boxes[i + 1] = temp

characters = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
# make_dirs(characters, r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR-CLASSIFY\characters")

character_dict = {}
for char in characters:
    character_dict[char] = 0

label_dict = {}
with open(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR-CLASSIFY\train.json", "r") as file:
    label_dict = json.load(file)

train_images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\images"
val_images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\val\images"
train_labels_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\labels"
val_labels_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\val\labels"
root_save_dir = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR-CLASSIFY\characters"

train_images = os.listdir(train_images_folder)
val_images = os.listdir(val_images_folder)
# train_labels = os.listdir(train_labels_folder)
# val_labels = os.listdir(val_labels_folder)

iteration = 0
for filename, label in label_dict.items():
    if iteration % 1000 == 0:
        print(iteration)
    
    iteration += 1

    image_file = filename + ".png"
    txt_file = filename + ".txt"
    # train_path = os.path.join(train_images_folder, image_file)
    val_path = os.path.join(val_images_folder, image_file)
    he_path = os.path.join(val_images_folder, "HE-"+image_file)
    inv_path = os.path.join(val_images_folder, "INV-"+image_file)
    he_inv_path = os.path.join(val_images_folder, "INV-HE-"+image_file)

    if os.path.exists(val_path):
        annotations = []
        current_label_path = os.path.join(val_labels_folder, txt_file)
        if not os.path.exists(current_label_path):
            continue
        with open(current_label_path, "r") as file:
            lines = file.readlines()
            annotations = [line.strip() for line in lines]
        image = cv2.imread(val_path)
        he_image = cv2.imread(he_path)
        inv_image = cv2.imread(inv_path)
        inv_he_image = cv2.imread(he_inv_path)
        boxes = [utils.annotation_to_points(image, ann) for ann in annotations]
        boxes.sort(key=lambda box: box[0])
        vertical_sort(boxes)

        # for box, char in zip(boxes, label):
        for box, char in zip(boxes, label):
            for padding in [1, 3, 5]:
                crop = utils.crop_from_points(image, box, padding)
                crop_he = utils.crop_from_points(he_image, box, padding)
                crop_inv = utils.crop_from_points(inv_image, box, padding)
                crop_inv_he = utils.crop_from_points(inv_he_image, box, padding)
                cv2.imwrite(os.path.join(root_save_dir, char, f"{char}-VAL-{character_dict[char]}.png"), crop)
                cv2.imwrite(os.path.join(root_save_dir, char, f"{char}-VAL-HE-{character_dict[char]}.png"), crop_he)
                cv2.imwrite(os.path.join(root_save_dir, char, f"{char}-VAL-INV-{character_dict[char]}.png"), crop_inv)
                cv2.imwrite(os.path.join(root_save_dir, char, f"{char}-VAL-INV-HE-{character_dict[char]}.png"), crop_inv_he)
                character_dict[char] += 1
        