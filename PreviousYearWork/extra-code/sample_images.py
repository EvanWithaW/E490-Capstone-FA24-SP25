import os
import shutil
from sklearn.utils.random import sample_without_replacement



images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\he-invhe-val\images"
labels_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\he-invhe-val\labels"
save_image_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\he-invhe-val\images-500"
save_label_folder =r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\he-invhe-val\labels-500"
images = os.listdir(images_folder)
# he = [image for image in images if "HE" in image and "INV" not in image]
inv_he = [image for image in images if "HE" in image and "INV" in image]

sample_indexes = sample_without_replacement(len(inv_he), 500, random_state=42)

for index in sample_indexes:
    image = inv_he[index]
    name, ext = os.path.splitext(image)
    txt = name + ".txt"
    shutil.copy2(os.path.join(images_folder, image), os.path.join(save_image_folder, image))
    shutil.copy2(os.path.join(labels_folder, txt), os.path.join(save_label_folder, txt))