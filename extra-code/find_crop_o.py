import os 
import shutil


def create_label_dict(file_lines):
    headers = file_lines[0].split(",")[:-1]
    data = file_lines[1:]
    label_dict = {}
    for line in data:
        UFM_ID,TXN_TIME,TOLLZONE_ID,LANE_POSITION,PLATE_TYPE,PLATE_TYPE_CONFIDENCE,PLATE_READ,PLATE_RDR_CONFIDENCE, \
        PLATE_JURISDICTION,IR_DISPOSITIONED,PAYMENT_METHOD,IMAGE1,IMAGE2,IMAGE3,IMAGE4,TYPE1,TYPE2,TYPE3,TYPE4 = line.split(",")[:-1]
        for image in [IMAGE1, IMAGE2, IMAGE3, IMAGE4]:
            if image != "None":
                label_dict[image.split(".")[0]] = PLATE_READ
    return label_dict


label_dict = {}
with open(os.path.join(r"D:\v2x-dataset", "data-11-30.csv"), "r") as file:
    label_dict = create_label_dict(file.readlines())

images_folder = r"D:\v2x-11-30-data\11-30-Parsed\ALL-CROPS"
save_folder = r"D:\v2x-11-30-data\11-30-Parsed\O-images"
images = os.listdir(images_folder)
o_count = 0

for image in images:
    name, ext = os.path.splitext(image)
    label = label_dict[name]

    if "O" in label:
        o_count += 1
        shutil.copy2(os.path.join(images_folder, image), os.path.join(save_folder, image))

print(o_count)