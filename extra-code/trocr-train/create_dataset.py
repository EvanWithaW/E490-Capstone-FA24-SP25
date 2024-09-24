import os
import shutil


def parse_csv(file_handle):
    lines = file_handle.readlines()
    headers = lines[0]
    data = [line.strip() for line in lines[1:]]
    num_records = len(data)

    return (headers, data, num_records)

train_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\reg-train\images"
val_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\reg-val\images"
save_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\trocr-train"
train_images = os.listdir(train_folder)
val_images = os.listdir(val_folder)


label_dict = {}
with open("CORRECT-LABELS.csv", "r") as file:
    headers, data, num_records = parse_csv(file)
    for line in data:
        label, image = line.split(",")
        label_dict[image] = label
    
keys = label_dict.keys()

with open("train-set.csv", "w") as file:
    file.write("text,file_name\n")
    for image_list, folder in zip([train_images, val_images], [train_folder, val_folder]):
        for image in image_list:
            name, ext = os.path.splitext(image)
            if name not in keys:
                continue
            file.write(f"{label_dict[name]},{name}\n")
            shutil.copy2(os.path.join(folder, image), os.path.join(save_folder, image))
    

