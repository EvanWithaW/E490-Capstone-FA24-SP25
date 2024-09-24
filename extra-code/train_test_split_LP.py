from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob
import shutil


def parse_csv(file_handle):
    lines = file_handle.readlines()
    headers = lines[0]
    data = lines[1:]
    num_records = len(data)

    return (headers, data, num_records)

def create_label_file():
    whole_list = []
    with open("all-train-test.txt", "r") as file:
        headers, data, num_records = parse_csv(file)
        special = 0
        for line in data:
            split_line = line.strip().split(",")
            label, pred, image, is_special = split_line
            if is_special == "True":
                special += 1
                whole_list.append([pred, image])
            else:
                whole_list.append([label, image])

    with open("CORRECT-LABELS.csv", "w") as file:
        file.write(f"LABEL,IMAGE\n")
        for entry in whole_list:
            label, image = entry
            file.write(f"{label},{image}\n")

def create_X_y_labels(file_path):
    with open(file_path, "r") as file:
        headers, data, num_records = parse_csv(file)
        X = []
        y = []
        for line in data:
            split_line = line.strip().split(",")
            label, image = split_line   
            X.append(image)
            y.append(label)

        np_X = np.array(X)
        np_y = np.array(y)
        
        return np_X, np_y


def check_num_labels(folder_path):
    paths = glob.glob(os.path.join(folder_path, "*"))
    multiple = 0
    for path in paths:
        with open(path, "r") as file:
            if len(file.readlines()) > 1:
                multiple += 1
    print(multiple)


def copy_files(from_folder, to_folder):
    from_files = os.listdir(from_folder)
    for file in from_files:
        shutil.copy2(os.path.join(from_folder, file), os.path.join(to_folder, file))

# path = "CORRECT-LABELS.csv"
# X, y = create_X_y_labels(path)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=42)
# print(len(X_train))
# print(len(X_test))

images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\train-test-images"
labels_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\train-test-plate-labels"
train_images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\train\images"
train_labels_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\train\labels"
val_images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\val\images"
val_labels_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\val\labels"
test_images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\test\images"
test_labels_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\test\labels"
label_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\train-test-plate-labels"
X = np.array(os.listdir(images_folder))
y = np.array(os.listdir(images_folder))
X_val_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=42)
X_train, X_val, y1, y2 = train_test_split(X_val_train, X_val_train, test_size=0.10, shuffle=True, random_state=42)
# print(len(X_train), len(X_val))

# for image in X_train:
#     shutil.copy2(os.path.join(images_folder, image), os.path.join(train_folder, image))

# for image in X_val:
#     shutil.copy2(os.path.join(images_folder, image), os.path.join(val_folder, image))

# for image in X_test:
#     shutil.copy2(os.path.join(images_folder, image), os.path.join(test_folder, image))


# labels
train_images = os.listdir(train_images_folder)
val_images = os.listdir(val_images_folder)
test_images = os.listdir(test_images_folder)
print(len(train_images), len(val_images), len(test_images))


def copy_txt_files(image_files, from_folder, to_folder):
    for image in image_files:
        name, ext = os.path.splitext(image)
        txt_file = name + ".txt"
        txt_path = os.path.join(from_folder, txt_file)
        if os.path.exists(txt_path):
            shutil.copy2(txt_path, os.path.join(to_folder, txt_file))

# copy_txt_files(train_images, labels_folder, train_labels_folder)
# copy_txt_files(val_images, labels_folder, val_labels_folder)
# copy_txt_files(test_images, labels_folder, test_labels_folder)

# txt_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\NO-LP\has-lp"
# txt_files = os.listdir(txt_folder)
# num = 0
# for txt_file in txt_files:
#     name, ext = os.path.splitext(txt_file)
#     print(name)
#     # shutil.copy2(os.path.join(txt_folder, txt_file), os.path.join(train_labels_folder, txt_file))
#     shutil.copy2(os.path.join(r"D:\v2x-11-30-data\ALPRPlateExport11-30-23", name+".jpg"), os.path.join(train_images_folder, name+".jpg"))
# print(num)




# /N/slate/jdmckean

