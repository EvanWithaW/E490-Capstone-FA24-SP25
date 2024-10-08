import re
import os
import json
import openpyxl
import scripts.utility as utils


annotation_folder = r"C:\Users\Jed\Desktop\v2x-dataset\kp-annotations"
trocr_data_path = os.path.join(r"C:\Users\Jed\Desktop\v2x-dataset", "trocr_results.json")
annotation_files = os.listdir(os.path.join(annotation_folder))
workbook_path = os.path.join(r"C:\Users\Jed\Desktop\v2x-dataset", "TrOCR_Test1.xlsx")
annotations = []
trocr_data = ""
correct = 0
incorrect = 0

# workbook = utils.create_init_workbook("TrOCR Results", ["Label", "OCR Output", "Image"])
workbook = openpyxl.load_workbook(filename=workbook_path)
sheet = workbook.active

with open(trocr_data_path, "r") as file:
    trocr_data = json.load(file)

for annotation_file in annotation_files:
    annotation_path = os.path.join(annotation_folder, annotation_file)
    annotation_data = utils.extract_from_datumaro(annotation_path)
    annotations += annotation_data

for image, ocr_value in trocr_data.items():
    for annotation in annotations:
        image_name, plate_num, points = annotation
        image_name = image_name[:-4]

        if image == image_name:
            ocr_value = re.sub('[\W_]+', '', ocr_value)
            if ocr_value == plate_num:
                correct += 1
            else:
                incorrect += 1
                row = [plate_num, ocr_value, f"{image}.jpg"]
                sheet.append(row)
                # print(f"OCR:{ocr_value}\t\tAnnotated:{plate_num}\t\tImage:{image}")

num_images = len(trocr_data.items())
print(f"Correct: {correct}\tIncorrect:{incorrect}\tNumber of Images: {num_images}")
print(f"Percentage Correct: {(correct / num_images) * 100}%")

workbook.save(filename=workbook_path)