import os
import json
import scripts.utility as utils


annotation_folder = r"C:\Users\Jed\Desktop\v2x-dataset\cars_kp_annotations"
annotation_files = os.listdir(os.path.join(annotation_folder))
data_path = r"C:\Users\Jed\Desktop\v2x-dataset\parsed_data.json"
annotations = []
num_cars = 0
num_images = 0

for annotation_file in annotation_files:
    annotation_path = os.path.join(annotation_folder, annotation_file)
    annotation_data = utils.extract_from_datumaro(annotation_path)
    annotations += annotation_data

with open(data_path, "r") as file:
    data_dict = json.load(file)
    keys = list(data_dict.keys())


    for annotation in annotations:
        image_name, labeled_plate, points = annotation

        for key in keys:
            associated_images = key.split(",")

            if image_name in associated_images:
                print(key, keys.index(key))
                num_cars += 1
                num_images += len(associated_images)
                keys.pop(keys.index(key))
                print(len(keys))
                break

print(f"Number of Cars: {num_cars}")
print(f"Number of Images: {num_images}")