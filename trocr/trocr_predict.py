import os
import sys
import glob
import cv2
import time
import re
import torch
import json
import openpyxl
import scripts.utility as utils
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


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


if len(sys.argv) < 3:
    print("Error. Arguments: image_folder_path, save_file_path")
    sys.exit()

image_folder = sys.argv[1]
save_file_path = sys.argv[2]
save_file_name = "trocr-new-data.xlsx"
label_path = os.path.join(r"C:\Users\Jed\Desktop\capstone_project\v2x-dataset", "data-11-30.csv")
save_path = os.path.join(save_file_path, save_file_name)
images = glob.glob(os.path.join(image_folder, "*"))
pred_dict = {}
workbook = ""
sheet = ""
headers = ["Label", "OCR", "Image"]

# if not os.path.isfile(save_path):
#     workbook = utils.create_init_workbook("TrOCR and Labels", headers=headers)
# else:
#     workbook = openpyxl.load_workbook(save_path)

# sheet = workbook.active

file_lines = ""
with open(label_path, "r") as file:
    file_lines = file.readlines()
label_dict = create_label_dict(file_lines)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

model.to(device)

tic = time.time()
toc = 0
first_tic = time.time()
iterations = 0
# with open("../trocr-new-results.csv", "w") as file:
#     file.write("LABEL,OCR,IMAGE,\n")

# for single chars
plate_number = ""

for img in images:
    image = cv2.imread(img) 
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    generated_ids = model.generate(pixel_values)
    generated_ids = generated_ids.cpu()

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    parsed_ocr_value = re.sub('[\W_]+', '', generated_text)
    
    # for single chars
    plate_number += parsed_ocr_value
    
    # [:-4] slice is to remove the file extension
    # image_name = img.split("\\")[-1][:-4]
    # label = label_dict[image_name]
    # row = [label, parsed_ocr_value, image_name]
    # write_line = ",".join(row) + ",\n"
    # file.write(write_line)
    # sheet.append(row)

    # pred_dict[image_name] = generated_text

    iterations += 1

    if (iterations % 100) == 0:
        toc = time.time()
        print(f"Iterations: {iterations}\tTime: {toc - tic} seconds")
        tic = time.time()
        # workbook.save(filename=save_path)

last_toc = time.time()

# total_time = last_toc - first_tic
# n_images = len(images)
# average_time = total_time / n_images
# print(f"Total Time: {total_time / 60} minutes\tAverage Time: {average_time} seconds for {n_images} images.")
print(plate_number)


# workbook.save(filename=save_path)
# with open(save_path, "w") as file:
#     json.dump(pred_dict, file)
