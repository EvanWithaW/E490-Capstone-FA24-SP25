import math
import json
import cv2
import numpy as np
from openpyxl import Workbook


# TODO Add image boundary checking in annotation_to_points and get_bounding_box_data
# TODO Add vertical character sorting to get_bounding_box_data

def annotation_to_points(image, annotation_info):
    
    klass, x_center, y_center, box_width, box_height = annotation_info.split(' ')
    image_height, image_width, channels = image.shape
 
    x_center = float(x_center) * image_width
    y_center = float(y_center) * image_height
    box_width = float(box_width) * image_width
    box_height = float(box_height) * image_height

    xmin = math.floor(x_center - (box_width/2))
    ymin = math.floor(y_center - (box_height/2))
    xmax = math.ceil(x_center + (box_width/2))
    ymax = math.ceil(y_center + (box_height/2))

    return [xmin, ymin, xmax, ymax]


def box_to_annotation(box, image_width, image_height, class_number):
    xmin, ymin, xmax, ymax = box
    box_width = xmax - xmin
    box_height = ymax - ymin
    x_center = xmin + (box_width/2)
    y_center = ymin + (box_height/2)

    # normalize values
    box_width = box_width / image_width
    box_height = box_height / image_height
    x_center = x_center / image_width
    y_center = y_center / image_height

    return f"{class_number} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

    
def crop_from_yolo_annotation(image, annotation_info):
    """
        Uses data from a yolo format annotation file to return a cropped section of the input image.

        image: image to be cropped. This is an array like object
        annotation_info: yolo format information
            format: space separated string "<class number> <box x center> <box y center> <box width> <box height>"
                box x center, box y center, box width, box height are normalized values between 0 and 1

        return value: cropped image -> array like
    """

    xmin, ymin, xmax, ymax = annotation_to_points(image, annotation_info)

    return image[ymin:ymax, xmin:xmax]


def crop_from_points(image, bbox_points, padding=0):
    """
        Uses four bounding box points to return a cropped section from the input image.

        image: image to crop -> array like
        bbox_points: [xmin, ymin, xmax, ymax] -> list

        return: a cropped image -> array like
    """
    
    xmin, ymin, xmax, ymax = bbox_points
    width, height, channels = image.shape
    xmin = max(0, math.floor(xmin) - padding)
    ymin = max(0, math.floor(ymin) - padding)
    xmax = min(height, math.ceil(xmax) + padding)
    ymax = min(width, math.ceil(ymax) + padding)

    return image[int(ymin):int(ymax), int(xmin):int(xmax)]

def vertical_sort(boxes):
    num_boxes = len(boxes)

    for i in range(0, num_boxes - 1):
        box1 = boxes[i][0]
        box2 = boxes[i + 1][0]
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        box1_height = ymax1 - ymin1
        box2_height = ymax2 - ymin2

        # 0.25 to account for the possibility of vertical overlap
        if (ymin1 >= (ymax2 - 0.25*box2_height)):
            # print("ran")
            if xmax1 >= xmin2 and xmax1 <= xmax2:
                temp = boxes[i]
                boxes[i] = boxes[i + 1]
                boxes[i + 1] = temp

def get_highest_conf(boxes):
    highest_conf = ""
    highest_conf = boxes[0]
    for i in range(len(boxes) - 1):
        if boxes[i][1] < boxes[i + 1][1]:
            highest_conf = boxes[i + 1]

        return [highest_conf]

def get_bounding_box_data(model_prediction, image, padding=0, model="lp"):
    """
        Retrieves bounding box data from a YOLOv5 model prediction output.
        Optionally adds padding to the bounding boxes

        model_prediction: a YOLOv5 model prediction. Format: [[xmin, ymin, xmax, ymax, confidence, class number]]
        padding: optional parameter to add padding to the bounding box. This increases the size of the bounding box.

        return: boxes list with bounding box list, confidence, and class number per box
                [[bounding_box, confidence, class_number], ...]
                [[xmin, ymin, xmax, ymax], confidence, class_number], ...]
                len(boxes) >= 1
    """
    
    boxes = []

    # for each bounding box predicted in the image
    for box in model_prediction:
        # box: [xmin, ymin, xmax, ymax, confidence, class number]
        bounding_box = box[:4]
        confidence = box[4]
        class_number = box[5]       # only two for now: license plate or character

        width, height, channels = image.shape
        
        xmin = max(0, math.floor(bounding_box[0]) - padding)
        ymin = max(0, math.floor(bounding_box[1]) - padding)
        xmax = min(height, math.ceil(bounding_box[2]) + padding)
        ymax = min(width, math.ceil(bounding_box[3]) + padding)

        bounding_box = [[xmin, ymin, xmax, ymax], confidence, class_number]
        boxes.append(bounding_box)

    # sort horizontally and vertically
        if len(boxes) > 1:
            if model == "lp":
                boxes = get_highest_conf(boxes)
            else:
                boxes.sort(key=lambda box: box[0])
                vertical_sort(boxes)

    return boxes


def extract_from_datumaro(json_file, finished_items=None):
    f = open(json_file)
    json_dict = json.load(f)
    
    data = []
    items = json_dict["items"]

    if finished_items:
        items = items[:finished_items]

    for item in items:
        id = item["id"]
        image_file = f"{id.split('/')[-1]}.jpg"
        annotations = item["annotations"]
        plate_number = ""
        points = []
        
        # check for labeled images
        if annotations:
            attributes = annotations[0]["attributes"]
            plate_number = ""
            keys = attributes.keys()
            if "plate number" in keys:
                plate_number = attributes["plate number"]
            elif "Plate Number" in keys:
                plate_number = attributes["Plate Number"]

            pts = annotations[0]["points"]
            
            for i in range(0, 8, 2):
                points.append([pts[i], pts[i + 1]])

            ### sort points: [top left, top right, bottom left, bottom right]
            # sort by y coordinate
            points.sort(key=lambda point: point[1])
            
            # sort each half by x corrdinate
            top = points[:2]
            top.sort(key=lambda point: point[0])
            bottom = points[2:]
            bottom.sort(key=lambda point: point[0])

            points = top + bottom
            

        data.append([image_file, plate_number, points])
    
    return data


def get_min_max(keypoints):
    top_left, top_right, bottom_left, bottom_right = keypoints
    xmin = min(top_left[0], bottom_left[0])
    xmax = max(top_right[0], bottom_right[0])
    ymin = min(top_left[1], top_right[1])
    ymax = max(bottom_left[1], bottom_right[1])

    return [xmin, xmax, ymin, ymax]  


def get_transform_points(keypoints, padding=None):
    xmin, xmax, ymin, ymax = get_min_max(keypoints)
    box_width = xmax - xmin
    box_height = ymax - ymin
    
    # point order: top left, bottom left, bottom right, top right 
    dest_points = np.float32([[0, 0],
                              [0, box_height-1],
                              [box_width-1, box_height-1],
                              [box_width-1, 0]])

    return [dest_points, box_width, box_height]


def deskew(image, points):
    top_left, top_right, bottom_left, bottom_right = points 
    input_points = np.float32([top_left, bottom_left, bottom_right, top_right])
    dest_points, width, height = get_transform_points(points)

    M = cv2.getPerspectiveTransform(input_points, dest_points)
    deskewed = cv2.warpPerspective(image, M, (int(width), int(height)), flags=cv2.INTER_LINEAR)

    return deskewed


def visualize_annotations(image_path, box=None, keypoints=None, box_color=(0, 255, 0), point_color=(0, 0, 255)):
    # TODO Add box annotation plotting

    annotated = ""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2.namedWindow("Annotation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Annotation", 600, 600)
    
    if keypoints:
        xmin, xmax, ymin, ymax = get_min_max(keypoints)
        annotated = cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 3)

        for point in keypoints:
            x, y = point
            annotated = cv2.circle(image, (int(x), int(y)), radius=10, color=point_color, thickness=-1)


    cv2.imshow("Annotation", annotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def create_init_workbook(sheet_title, headers):
    workbook = Workbook()
    sheet = workbook.active
    headers = headers
    sheet.title = sheet_title
    sheet.append(headers)

    return workbook