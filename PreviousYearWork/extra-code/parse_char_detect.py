import os
import shutil


def parse_csv(file_handle):
    lines = file_handle.readlines()
    headers = lines[0]
    data = lines[1:]
    num_records = len(data)

    return (headers, data, num_records)

move_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\not-in"
train_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\images"
labels_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\labels"
test_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\test\images"
val_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\val\images"
val_label = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\val\labels"
test_labels = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\test\labels"

bad_images = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\bad-images"

train_images = os.listdir(test_folder)
labels = os.listdir(test_labels)

label_dict = {}
with open("CORRECT-LABELS.csv", "r") as file:
    headers, data, num_records = parse_csv(file)

    for entry in data:
        label, image = entry.strip().split(",")
        label_dict[image] = label

not_in = 0
keys = label_dict.keys()
for image in train_images:
    name, ext = os.path.splitext(image)

    if name not in keys and not name.isdigit():
        not_in += 1
        shutil.move(os.path.join(test_folder, image), os.path.join(move_folder, image))
        shutil.move(os.path.join(test_labels, name+".txt"), os.path.join(move_folder, name+".txt"))

print(not_in)

# for image in os.listdir(move_folder):
#     name,ext = os.path.splitext(image)
#     txt = name + ".txt"
#     print(txt)
#     txt_path = os.path.join(move_folder, txt)
#     if os.path.exists(txt_path):
#         shutil.move(txt_path, os.path.join(bad_images, txt))
