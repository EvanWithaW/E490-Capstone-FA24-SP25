import easyocr
import sys
import os
import glob
import csv


def easyocr_test(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image_path)
    for detection in result:
        if detection[2] > 0.5:
            # print(detection[1])
            return detection[1]
        
def easyocr_test_batch(image_folder):
    prediction_list = []
    reader = easyocr.Reader(['en'], gpu=False)
    images = glob.glob(os.path.join(image_folder, "*"))
    fields = ["PREDICTION", "CONFIDENCE", "IMAGE"]

    with open("easyocr-test-set-cropped.csv", "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        
        for image in images:
            image_name = image.split("\\")[-1]
            
            allow_list = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            result = reader.readtext(image, allowlist=allow_list, )#rotation_info=[90,180,270])
            for detection in result:
                read = detection[1]
                confidence = detection[2]
                row = [read, confidence, image_name]
                csvwriter.writerow(row)


if __name__ == '__main__':
    image_path = sys.argv[1]
    # easyocr_test(image_path)
    easyocr_test_batch(image_path)

