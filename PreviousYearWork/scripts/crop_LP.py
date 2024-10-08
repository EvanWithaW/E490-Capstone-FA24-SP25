import cv2
import glob
import os
import sys
from utility import crop_from_yolo_annotation
    

"""
    This script takes an image folder, yolo annotations folder, and an ouput folder.
    Images are cropped using yolo annotation format and stored in the ouput folder.

    argument 1: image folder
    argument 2: annotations folder
    argument 3: output folder
"""

def main():
    if len(sys.argv) != 4:
        print("Incorrect number of arguments.\nArguments: image_folder annotation_folder output_folder")
        sys.exit()

    image_folder = sys.argv[1]
    annotations_folder = sys.argv[2]
    output_folder = sys.argv[3]

    images = os.listdir(image_folder)
    
    for img in images:
        ann_file = img.split(".")[0] + ".txt"
        annotation_path = os.path.join(annotations_folder, ann_file)
        image_path = os.path.join(image_folder, img)
        if os.path.exists(annotation_path):
            with open(annotation_path) as f:
                image = cv2.imread(image_path)
                image_name, image_extension = img.split(".")
                info = f.readline()
                
                if info == "":
                    continue
                
                cropped_image = crop_from_yolo_annotation(image, info)
                if cropped_image.size == 0:
                    continue
                save_path = os.path.join(output_folder, f"{image_name}.png")
                # cv2.imwrite(save_path, cropped_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                cv2.imwrite(save_path, cropped_image)


if __name__ == "__main__":
    main()


