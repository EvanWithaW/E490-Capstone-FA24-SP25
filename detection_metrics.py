import torch
import os
import cv2
import scripts.utility as utils
from torchvision import ops
import numpy as np
import math
import sys


def get_iou(prediction_box, ground_truth_box):
    """
        ** TODO: ADD DOCSTRING**
    """
    truth_box_tensor = torch.tensor([ground_truth_box], dtype=torch.float)
    prediction_box_tensor = torch.tensor([prediction_box], dtype=torch.float)

    iou = ops.box_iou(truth_box_tensor, prediction_box_tensor)

    return iou.numpy()[0][0].item()

def character_detect_metrics(pred_folder, label_folder, images_folder):
    labels = os.listdir(label_folder)
    predictions = os.listdir(pred_folder)
    images = os.listdir(images_folder)

    # 17584 for all three
    print(len(images))
    print(len(labels))
    print(len(predictions))

    eq_chars = 0
    diff_chars = 0
    total_chars = 0
    correct_chars = 0
    threshold = 0.8

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for label in labels:
        name, ext = os.path.splitext(label)
        image = cv2.imread(os.path.join(images_folder, name + ".png"))
        with open(os.path.join(label_folder, label), "r") as label_file, open(os.path.join(pred_folder, label), "r") as pred_file:
            label_lines = [l.strip() for l in label_file.readlines()]
            pred_lines = [l.strip() for l in pred_file.readlines()] 
            total_chars += len(label_lines)

            label_len = len(label_lines)
            pred_len = len(pred_lines)
            
            if label_len == pred_len:
                eq_chars += 1
                label_boxs = [utils.annotation_to_points(image, l) for l in label_lines]
                pred_boxs = [utils.annotation_to_points(image, p) for p in pred_lines]
                ious = [get_iou(pred_box, label_box) for pred_box, label_box in zip(pred_boxs, label_boxs)]
                correct_incorrect = [1 for iou in ious if iou > threshold]
                total_true = sum(correct_incorrect)
                correct_chars += total_true
                TP += total_true
                FP += len(ious) - total_true
            else:
                diff_chars += 1
                if label_len > pred_len:
                    FN += label_len - pred_len
                elif label_len < pred_len:
                    FP += pred_len - label_len
    
    return correct_chars, total_chars, eq_chars, diff_chars, TP, FP, TN, FN


def lp_detect_metrics(pred_folder, label_folder, images_folder):
    labels = os.listdir(label_folder)
    predictions = os.listdir(pred_folder)
    images = os.listdir(images_folder)

    
    print(len(images))
    print(len(labels))
    print(len(predictions))
    len_labels = len(labels)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    threshold = 0.8
    multiple_plates = 0

    preds = [file for file in predictions if file in labels]
    preds2 = [file for file in predictions if file not in labels]
    print(len(preds2))
    return TP, FP, TN, FN
    for i, label in enumerate(labels):
        if i % 100 == 0:
            print(f"{i}/{len(labels)}")
        name, ext = os.path.splitext(label)
        image = cv2.imread(os.path.join(images_folder, name + ".jpg"))
        with open(os.path.join(label_folder, label), "r") as label_file, open(os.path.join(pred_folder, label)) as pred_file:
            label_lines = label_file.readlines()
            pred_lines = pred_file.readlines()
            
            if len(pred_lines) > 1:
                multiple_plates += 1
                continue

            # if len(label_lines) != len(pred_lines):
            else:
                label_line = label_lines[0]
                pred_line = pred_lines[0]
                label_box = utils.annotation_to_points(image, label_line)
                pred_box = utils.annotation_to_points(image, pred_line)
                iou = get_iou(pred_box, label_box)
                
                if iou > threshold:
                    TP += 1
                else:
                    FP += 1

        # print(f"TP: {TP}\tFP: {FP}\tTN: {TN}\tFN: {FN}")
    FP += multiple_plates
    return TP, FP, TN, FN


### License Plates
pred_folder = r"D:\v2x-11-30-data\TEST-LABELS\LP-DETECT"
label_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\test\labels"
images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-LP\test\images"
TP, FP, TN, FN = lp_detect_metrics(pred_folder, label_folder, images_folder)
print(f"TP: {TP}\tFP: {FP}\tTN: {TN}\tFN: {FN}")

### Characters
# pred_folder = r"D:\v2x-11-30-data\TEST-LABELS\CHAR-DETECT-INV-HE"
# label_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\test\labels"
# images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\test-he-inv"
# correct_chars, total_chars, eq_chars, diff_chars, TP, FP, TN, FN = character_detect_metrics(pred_folder, label_folder, images_folder)            
# print(f"Equal Characters: {eq_chars}\tDifferent Characters: {diff_chars}")
# print("Correct Characters: ", correct_chars)
# print("Total Characters: ", total_chars)
# print(f"TP: {TP}\tFP: {FP}\tTN: {TN}\tFN: {FN}")