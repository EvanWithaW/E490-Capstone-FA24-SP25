import os
import glob
import shutil


def parse_csv(file_handle):
    lines = file_handle.readlines()
    headers = lines[0]
    data = lines[1:]
    num_records = len(data)

    return (headers, data, num_records)

images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\images"
move_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\check"
ann_paths = glob.glob(os.path.join(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\labels", "*"))
label_dict = {}
with open("CORRECT-LABELS.csv", "r") as file:
    headers, data, num_records = parse_csv(file)

    for entry in data:
        label, image = entry.strip().split(",")
        label_dict[image] = label

incorrect = 0
not_in = 0
for path in ann_paths:
    is_not_in = False
    base, ann_file = os.path.split(path)
    image, ext = os.path.splitext(ann_file)
    with open(path, "r") as file:
        num_chars = len(file.readlines())
        if image not in label_dict.keys():
            not_in += 1
            is_not_in = True
        elif num_chars != len(label_dict[image]):
            incorrect += 1

    if is_not_in:
        shutil.move(path, os.path.join(move_folder, ann_file))
        shutil.move(os.path.join(images_folder, image+".png"), os.path.join(move_folder, image+".png"))

print(incorrect)
print(not_in)
