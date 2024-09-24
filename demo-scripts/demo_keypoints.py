import os
import cv2
import scripts.utility as utils


annotation_path = r"C:\Users\Jed\Desktop\kp_annotations.json"
annotations = utils.extract_from_datumaro(annotation_path, 608)

image_folder = r"C:\Users\Jed\Desktop\v2x-dataset\cap-images"

for annotation in annotations[0:10]:
    image_name, plate_number, points = annotation

    if not points: continue

    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    deskewed = utils.deskew(image, points)
    xmin, xmax, ymin, ymax = utils.get_min_max(points)
    cropped = image[int(ymin):int(ymax), int(xmin):int(xmax)]

    cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Perspective Corrected", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Cropped", 300, 100)
    cv2.resizeWindow("Perspective Corrected", 300, 100)

    utils.visualize_annotations(image_path, keypoints=points)
    cv2.imshow("Cropped", cropped)
    cv2.imshow("Perspective Corrected", deskewed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



