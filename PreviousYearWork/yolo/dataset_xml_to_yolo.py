import xml.etree.ElementTree as ET
import os
import glob
import cv2
import math

def xml_to_yolo_annotations(folder):
    output_dir = 'kaggle_LP_dataset_yolo/annotations'
    files = glob.glob(os.path.join(folder, '*.xml'))

    for file in files:
        tree = ET.parse(file)
        root = tree.getroot()

        save_filename = f"{root.find('filename').text.split('.')[0]}.txt"
        image_width = int(root.find('size').find('width').text)
        image_height = int(root.find('size').find('height').text)

        bounding_box = root.find('object').find('bndbox')
        xmin = int(bounding_box.find('xmin').text)
        xmax = int(bounding_box.find('xmax').text)
        ymin = int(bounding_box.find('ymin').text)
        ymax = int(bounding_box.find('ymax').text)

        x_center = xmin + int((xmax-xmin)/2)
        y_center = ymin + int((ymax-ymin)/2)
        x_center_norm = x_center/image_width
        y_center_norm = y_center/image_height

        box_width_norm = (xmax - xmin) / image_width
        box_height_norm = (ymax - ymin) / image_height

        
        # print(save_filename, image_width, image_height, (xmin, xmax, ymin, ymax))
        # print(f"Filename: {save_filename} image width: {image_width} image height: {image_height} (xmin, xmax, ymin, ymax): ({xmin}, {xmax}, {ymin}, {ymax})")
        # print(f"Normalized x center: {x_center_norm} Normalized y center: {y_center_norm}")
        # print(f"Normalized Box Width: {box_width_norm} Normalized Box Height: {box_height_norm}")

        with open(os.path.join(output_dir, save_filename), 'w', encoding='utf-8') as f:
            f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}")


def draw_bounding_boxes(image_folder, annotations_folder, output_image_folder):
    num = 0
    image_files = glob.glob(os.path.join(image_folder, '*.*'))
    annotation_files = glob.glob(os.path.join(annotations_folder, "*.txt"))
    for im, an in zip(image_files, annotation_files):

        image = cv2.imread(im)
        annotation_file = an
        info = ''
        start = 0
        end = 0
        height, width, channels = image.shape   #(rows, cols, channels)
        klass = x_center = y_center = box_width = box_height = 0

        with open(annotation_file) as f:
            info = f.readline()
            klass, x_center, y_center, box_width, box_height = info.split(' ')
            
            # denormalize centers and box dimensions
            x_center = int(float(x_center) * width)
            y_center = int(float(y_center) * height)
            box_width = int(float(box_width) * width)
            box_height = int(float(box_height) * height)

            # calculate top left point and bottom right point
            xmin = x_center - math.ceil(box_width/2)
            xmax = x_center + math.ceil(box_width/2)
            ymin = y_center - math.ceil(box_height/2)
            ymax = y_center + math.ceil(box_height/2)

            start = (xmin, ymin)
            end = (xmax, ymax)

            # color = (BGR)
            cv2.rectangle(image, start, end, color=(170, 255, 0), thickness=2)
            cv2.imwrite(os.path.join(output_image_folder, f"Cars{num}.png"), image)
        num += 1

    # cv2.imshow("image window", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()





#xml_to_yolo_annotations('kaggle_LP_dataset/annotations')

#draw_bounding_boxes('kaggle_LP_dataset_yolo/images', 'kaggle_LP_dataset_yolo/annotations', 'annotated_images')
