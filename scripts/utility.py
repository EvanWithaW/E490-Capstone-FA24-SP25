import math
import json
import cv2
import numpy as np
import re
import torch


##############################################
# Bounding Box and annotations handling functions
##############################################
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


def crop_from_points(image, bbox_points):
    """
        Uses four bounding box points to return a cropped section from the input image.

        image: image to crop -> array like
        bbox_points: [xmin, ymin, xmax, ymax] -> list

        return: a cropped image -> array like
    """
    
    xmin, ymin, xmax, ymax = bbox_points

    return image[int(ymin):int(ymax), int(xmin):int(xmax)]

def vertical_sort(boxes):
    """
        boxes: output from a get_bounding_box_data function call

        - A horizontal sort has already been done with respect to the top left point of each box and the x-axis
    """
    num_boxes = len(boxes)

    for i in range(0, num_boxes - 1):
        box1 = boxes[i][0]
        box2 = boxes[i + 1][0]
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        box1_height = ymax1 - ymin1
        box2_height = ymax2 - ymin2

        # check that the box is above the next box
        # 0.25 to account for the possibility of vertical overlap between boxes
        if (ymin1 >= (ymax2 - 0.25*box2_height)):
            # check if the first box needs to be moved
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
        image: an opencv loaded image(numpy array)
        padding: optional parameter to add padding to the bounding box. This increases the size of the bounding box.
        model: specifies the model type that produced the model_prediction
            Side effects:
                model="LP" -> takes the highest confidence bounding box
                model="char" -> sorts the character bounding boxes horizontally and vertically

        return: boxes list with bounding box list, confidence, and class number per box
                [[bounding_box, confidence, class_number], ...]
                [[xmin, ymin, xmax, ymax], confidence, class_number], ...]
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

    
        if len(boxes) > 1:
            # take the box with the highest confidence
            if model == "lp":
                boxes = get_highest_conf(boxes)
            # sort horizontally and vertically
            elif model == "char":
                boxes.sort(key=lambda box: box[0])
                vertical_sort(boxes)

    return boxes

##############################################
# Image Processing functions
##############################################
def HE(img):
    """
        img: opencv loaded image(numpy array)

        Returns a histogram equalized image
    """
    
    # Convert the image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)

    # Apply histogram equalization to the L channel
    l_equalized = cv2.equalizeHist(l)

    # Merge the equalized L channel with the original A and B channels
    lab_equalized = cv2.merge((l_equalized, a, b))

    # Convert the equalized LAB image back to BGR color space
    equalized_img = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

    return equalized_img

##############################################
# Model output handling and prediction functions
##############################################
def get_crop_lp(model_output, image):
    """
        Returns either a license plate with the highest confidence score
            or an empty numpy array
    """
    crop = np.array([])
    for prediction in model_output.xyxy:
        if prediction.numel() == 0:    
            continue

        prediction = prediction.tolist()     
        boxes = get_bounding_box_data(prediction, image, padding=0, model="lp")

        for box in boxes:
            bbox, conf, klass = box
            crop = crop_from_points(image, bbox)

    return crop

def get_crops_chars(model_output, image):
    """
        Returns a list of sorted character image crops
    """
    char_list = []
    for prediction in model_output.xyxy:
        if prediction.numel() == 0:    
            continue

        prediction = prediction.tolist()    
        
        boxes = get_bounding_box_data(prediction, image, padding=1, model="char")
        for box in boxes:
            bbox, conf, klass = box
            crop = crop_from_points(image, bbox)
            char_list.append(crop)

    return char_list 


def predict_chars(character_crops, classifier, transforms, device):
    """
        character_crops: a list of cropped character images
        classifier: image classification model(Resnet50)
        transforms: albumations image transforms for letterboxing and image normalization
        device: torch device to send the image to for model processing: cuda(gpu) or cpu
    """
    labels = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ" 
    pred_str = ""
    with torch.no_grad():
        for char_image in character_crops:
            char_image = image = transforms(image=char_image)["image"]
            char_image = char_image.unsqueeze(0).to(device)
            char_pred = classifier(char_image)
            values, real = torch.max(char_pred, 1)
            pred_str += labels[real]

    return pred_str

def pred_lp_trocr(lp_crop, trocr_model, processor, device):
    """
        lp_crop: a cropped license plate image
        trocr_model: hugging face TrOCR Model
        processor: Hugging face TrOCR prosessor 
        device: torch device to send the image to for model processing: cuda(gpu) or cpu
    """
    pixel_values = processor(images=lp_crop, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    generated_ids = trocr_model.generate(pixel_values)
    generated_ids = generated_ids.cpu()
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # regex to remove whitespace and extra characters that are not numbers or letters
    #   return just the generated text above if you don't want the extra stuff parsed out
    parsed_ocr_value = re.sub('[\W_]+', '', generated_text[0])

    # this joins on whitespace and keeps extra characters
    # lp_prediction = "".join(generated_text) 

    return parsed_ocr_value

##############################################
# Keypoint plate detector helper functions
##############################################
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


def extract_from_datumaro(json_file, finished_items=None):
    """
        Extract Plate data from exported CVAT
            datumaro file formatted file
    """
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


##############################################
# File handling functions
##############################################
def parse_csv(file_handle):
    lines = file_handle.readlines()
    headers = lines[0]
    data = lines[1:]
    num_records = len(data)

    return (headers, data, num_records)

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
