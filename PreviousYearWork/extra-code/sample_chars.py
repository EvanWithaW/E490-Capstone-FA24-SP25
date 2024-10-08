import os
import shutil
from sklearn.utils.random import sample_without_replacement


maximum = 30000
save_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR-CLASSIFY\train"
characters = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
for char in characters:
    num_chars = 0
    folder = os.path.join(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR-CLASSIFY\characters", char)
    images = os.listdir(folder)
    real = [image for image in images if "aug" not in image and "INV" not in image and "HE" not in image]
    he = [image for image in images if "HE" in image and "INV" not in image]
    inv_he = [image for image in images if "HE" in image and "INV" in image]
    len_real = min(len(real), 10000)
    len_he = min(len(he), 10000)
    len_inv_he = min(len(inv_he), 10000)

    print(char, len_real, len_he, len_inv_he)
    number_augs = maximum - len_real - len_he - len_inv_he

    for image_list, len_sample in zip([real, he, inv_he], [len_real, len_he, len_inv_he]):
        sample_indexes = sample_without_replacement(len(image_list), len_sample, random_state=42)
        for index in sample_indexes:
            shutil.copy2(os.path.join(folder, image_list[index]), os.path.join(save_folder, image_list[index]))
            num_chars += 1

    if number_augs > 0:
        aug = [image for image in images if "aug" in image]
        sample_indexes = sample_without_replacement(len(aug), number_augs)
        for index in sample_indexes:
            if num_chars >= maximum:
                break
            shutil.copy2(os.path.join(folder, aug[index]), os.path.join(save_folder, aug[index]))
            num_chars += 1


