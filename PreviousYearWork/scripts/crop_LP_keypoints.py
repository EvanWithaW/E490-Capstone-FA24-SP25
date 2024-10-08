import os
import cv2
import sys
import scripts.utility as utils


if len(sys.argv) < 4:
    print("Error. Parameters: images_folder annotation_path, save_folder")
    sys.exit()

image_folder = sys.argv[1]
annotations_path = sys.argv[2]
save_folder = sys.argv[3]
annotations = utils.extract_from_datumaro(annotations_path)

for annotation in annotations:
    image_name, plate_number, points = annotation
    save_image = image_name.split(".")[0] + ".png"
    image_path = os.path.join(image_folder, image_name)
    save_path = os.path.join(save_folder, save_image)

    if not points: continue 

    image = cv2.imread(image_path)
    deskewed = utils.deskew(image, points)
    cv2.imwrite(save_path, deskewed, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
