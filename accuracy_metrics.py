import shutil
import os
import sys
import json
import scripts.utility as utils
from difflib import SequenceMatcher as SM


def check_for_special(label, pred):
    """
        Checks for additional characters added to the license plate number label
    """
        ### Added checking:
    # prepend: 
    #   CA(plate type: AHNR) 
    #   PV,TP,MWRT(on plate) (plate type: ATNA)
    #   -(minus sign) in the plate number somewhere (plate type: IPASS, MO,)
    #   SB(plate type: SBNR)
    #   SM(plate type: SMNR)
    #   ST: STATE(on plate)(plate type: STNS)
    #   SP: (plate type: STNS)
    #   contains a . (plate type: 56)
    #   AMB (Plate type AMNR)
    #   BU (plate type BUNR)
    special = ["BU", "TA", "VT", "PH", "HR", "MB", "CA", "PV", "TP", "SB", "SM", "ST", "SP", ".", "-", "+", "&", "AMB"]
    correct_pred = False

    if label[:3] == "AMB" and (len(pred)+3 == len(label)):
        
        if pred == label[3:]:
            correct_pred = True
    elif label[:2] in special and (len(pred)+2 == len(label)):
        if pred == label[2:]:
            correct_pred = True
    elif label[0] == "&":
        if pred == label[1:-1]:
            correct_pred = True
    elif "." in label:
        if label.replace(".", "") == pred:
            correct_pred = True
    elif "-" in label:
        if label.replace("-", "") == pred:
            correct_pred = True
    elif "+" in label:
        if label.replace("+", "") == pred:
            correct_pred = True
    
    return correct_pred


def check_results_capstone(prediction_path):
    correct = 0
    incorrect = 0
    num_records = 0
    different_length = 0
    incorrect_but_train = 0

    with open(prediction_path, "r") as file:
        headers, data, num_records = utils.parse_csv(file)

        for line in data:
            label = ""
            image = ""
            pred = ""
            split_line = line.strip().split(",")
            label, PRED_REG, PRED_INV, PRED_HE, PRED_INV_HE, image = split_line
            pred = PRED_HE
            
            if check_for_special(label, pred):
                correct += 1
            else:         
                if pred == label:
                    correct += 1
                else:
                    incorrect += 1
                    percent_incorrect = SM(None, label, pred).ratio() * 100
                    if (len(pred) == len(label)) and (percent_incorrect > 70):
                        incorrect_but_train += 1
                    elif (len(pred) != len(label)):
                        different_length += 1
    return (correct, incorrect, num_records, different_length, incorrect_but_train)


def check_results(results_path, label_dict):
    correct = 0
    incorrect = 0
    num_records = 0
    different_length = 0
    incorrect_but_train = 0
    with open(results_path, "r") as file:
        headers, data, num_records = utils.parse_csv(file)
        for line in data:
            label = ""
            image = ""
            pred = ""
            split_line = line.strip().split(",")
            pred, image = split_line
            label = label_dict[image]

            if check_for_special(label, pred):
                correct += 1
            else:         
                if pred == label:
                    correct += 1
                else:
                    incorrect += 1
                    percent_incorrect = SM(None, label, pred).ratio() * 100
                    if (len(pred) == len(label)) and (percent_incorrect > 70):
                        incorrect_but_train += 1
                    elif (len(pred) != len(label)):
                        different_length += 1
    return (correct, incorrect, num_records, different_length, incorrect_but_train)

                    
if len(sys.argv) < 2:
    print("Incorrect Arguments. Provide either CAPSTONE for capstone results or a path for a runs file")
    sys.exit()
test_type = sys.argv[1]
### To reproduce final capstone results: 
#       Replace the two file paths below with correct path location
if test_type == "CAPSTONE":
    # capstone_full_path = r"C:\Users\Jed\Desktop\capstone_project\4-25-full-results.csv"
    # capstone_small_path = r"C:\Users\Jed\Desktop\capstone_project\4-25-small-results.csv"
    capstone_full_path = sys.argv[2]
    capstone_small_path = sys.argv[3]

    if not os.path.exists(capstone_full_path) or not os.path.exists(capstone_small_path):
        print("Capstone file paths incorrect or the files do not exist.")
        sys.exit()

    correct_full, incorrect_full, num_records_full, different_length_full, incorrect_but_train_full = check_results_capstone(capstone_full_path)
    correct_small, incorrect_small, num_records_small, different_length_small, incorrect_but_train_small = check_results_capstone(capstone_small_path)

    print("CAPSTONE FINAL TEST RESULTS")
    print("Full Test")
    print("------------------------")
    print(f"Correct: {correct_full}/{num_records_full}\t{(correct_full / num_records_full) * 100 :0.5f}%")
    print("Incorrect: ", incorrect_full)
    print("Different Length: ", different_length_full)
    # print("Incorrect but train: ", incorrect_but_train_full)
    print("------------------------")

    print()
    print("Small Test")
    print("------------------------")
    print(f"Correct: {correct_small}/{num_records_small}\t{(correct_small / num_records_small) * 100 :0.5f}%")
    print("Incorrect: ", incorrect_small)
    print("Different Length: ", different_length_small)
    # print("Incorrect but train: ", incorrect_but_train_small)
    print("------------------------")
    print()

else:
    results_path = sys.argv[1]
    if not os.path.exists(results_path):
        print("Results file path incorrect.")
        sys.exit()
    label_dict = {}
    label_file_path = "data-11-30.csv"
    with open(label_file_path, "r") as file:
        label_dict = utils.create_label_dict(file.readlines())

    correct, incorrect, num_records, different_length, incorrect_but_train = check_results(results_path, label_dict)
    
    print("Results")
    print("------------------------")
    print(f"Percentage chars correct: {(correct / num_records) * 100 :0.5f}%")
    print(f"{correct}/{num_records}")
    print()
    print("Incorrect: ", incorrect)
    print("Different Length: ", different_length)
    print("------------------------")
    # print("Incorrect but train: ", incorrect_but_train)
