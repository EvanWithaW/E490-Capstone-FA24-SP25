import os
import json


def parse_csv(file_handle):
    lines = file_handle.readlines()
    headers = lines[0]
    data = lines[1:]
    num_records = len(data)

    return (headers, data, num_records)


label_dict = {}
with open("CORRECT-LABELS.csv", "r") as file:
    headers, data, num_records = parse_csv(file)

    for entry in data:
        label, image = entry.strip().split(",")
        label_dict[image] = label

test_images = []
with open(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\test\test-images.txt", "r") as file:
    lines = file.readlines()
    test_images = [line.strip().split(".")[0] for line in lines]


train_images = {}
for key, value in label_dict.items():
    if key not in test_images:
        train_images[key] = value

with open(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR-CLASSIFY\train.json", "w") as file:
    json.dump(train_images, file)