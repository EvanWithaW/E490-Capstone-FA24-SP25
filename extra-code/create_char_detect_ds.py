import os


def parse_csv(file_handle):
    lines = file_handle.readlines()
    headers = lines[0]
    data = lines[1:]
    num_records = len(data)

    return (headers, data, num_records)


train_txt_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\train\labels"
test_txt_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\test\labels"
val_txt_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\val\labels"

train_images_folder =r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\train\images" 
val_images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\val\images"
test_images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\test\images"
multiple_images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\MULTIPLE-PLATES-WHOLE\obj_train_data"

train_images = os.listdir(train_images_folder)
val_images = os.listdir(val_images_folder)
test_images = os.listdir(test_images_folder)
no_lp_images = os.listdir(r"D:\v2x-11-30-data\11-30-Parsed\no_lp")
multiple_files = os.listdir(multiple_images_folder)
multiple_images = [image for image in multiple_files if image.endswith(".jpg")]

train_images_no_multiple = [image for image in train_images if image not in multiple_images and image not in no_lp_images]
val_images_no_mult = [image for image in val_images if image not in multiple_images and image not in no_lp_images]
test_images_no_mult = [image for image in test_images if image not in multiple_images and image not in no_lp_images]
print(len(train_images_no_multiple), len(val_images_no_mult), len(test_images_no_mult))

# with open(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\train-images.txt", "w") as file:
#     for image in train_images_no_multiple:
#         file.write(image+"\n")

with open(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\test\test-images.txt", "w") as file:
    for image in test_images_no_mult:
        file.write(image+"\n")

with open(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\val\val-images.txt", "w") as file:
    for image in val_images_no_mult:
        file.write(image+"\n")
