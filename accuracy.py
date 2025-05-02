import sys
import pandas as pd

label_path = sys.argv[1]
results_path = sys.argv[2]
confidence = int(sys.argv[3])

label_df = pd.read_csv(label_path)
results_df = pd.read_csv(results_path)

def extract_image_name(view):
    image = view.rsplit('/')[-1]
    return image.split('.jpg')[0]

label_df = pd.read_csv(label_path, low_memory=False)
results_df = pd.read_csv(results_path)

# Ensure UFM_ID is unique by dropping duplicates
label_df = label_df.drop_duplicates(subset="UFM_ID")

 # create a label dictionary based on the csv (ALPRPlateExport11-30-23.csv) with only the plate read info and various image angles
ufm_dict = label_df.set_index("UFM_ID")[["PLATE_READ", "IMAGE1", "IMAGE2", "IMAGE3", "IMAGE4"]].to_dict(orient="index")
# create a dictionary based on the results of our model's predictions containing the image name, the prediction, and the confidence
results_dict = {row["IMAGE"]: (row["PRED"], row["CONF"]) for _, row in results_df.iterrows()}

# rename the image values in the ufm dict to only contain the relevant part 
# (e.g. http://matip05e.massaets.com/TxnViewer/images/20231127/R0010/C0069/090005_1701090002939F02_331.jpg -> 0py90005_1701090002939F02_331.jpg)
for key, val in ufm_dict.items():
    for key2, val2 in val.items():
        if (key2.startswith("IMAGE") and type(val2) == str):
            val[key2] = extract_image_name(val2)

# create a dict that contains the ufm id as the key and a tuple of the image id (IMAGE1, IMAGE2, IMAGE3, or IMAGE4 for each ufm id), prediction, and confidence as vals
sorted_results_dict = {}
for ufm_id, vals in ufm_dict.items():  # val is image1: name, image2: name, etc. and plate read
    for key2, val2 in vals.items():
        if key2.startswith("IMAGE"):
            image_id = vals[key2]
            
            # check to see if the image id is present in the results dict before appending it to the final dict
            if image_id in results_dict:
                pred, conf = results_dict[image_id]

                # populate the sorted dict with ufm id as the key and tuple of the image id, prediction, and confidence as vals
                if ufm_id not in sorted_results_dict or conf > sorted_results_dict[ufm_id][2]:
                    sorted_results_dict[ufm_id] = (image_id, pred, conf)

tp, fp, tn, fn = 0, 0, 0, 0
manual, auto = 0, 0
missing, present = 0, 0
for ufm_id, vals in ufm_dict.items():
    if ufm_id not in sorted_results_dict:
        missing += 1
        continue
    else:
        present += 1
    lp_label = vals["PLATE_READ"]
    lp_pred = sorted_results_dict[ufm_id][1]
    lp_conf = sorted_results_dict[ufm_id][2]
    if lp_conf < confidence:
        manual += 1
        if lp_label == lp_pred:
            fn += 1
        else:
            tn += 1
    elif lp_label == lp_pred:  # confidence >= 900
        auto += 1
        tp += 1
    else:
        auto += 1
        fp += 1

print(f"\nPrecision: {tp / (tp + fp):.4f}")
print(f"Automation Rate: {auto / (auto + manual+1e-6):.4f}")
print("")
print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn+1e-6):.4f}")
print(f"Recall: {tp / (tp + fn+1e-6):.4f}")
print(f"F1 Score: {2 * (tp / (tp + fp+1e-6)) * (tp / (tp + fn+1e-6)) / ((tp / (tp + fp+1e-6)) + (tp / (tp + fn+1e-6))+1e-6):.4f}")
print(f"Automatic Plates Read: {auto}")
print(f"Manual Plates Read: {manual}")
print(f"Total Images: {auto+manual}")
print(f"TP TN FP FN: {tp} {tn} {fp} {fn}")
print("")
# print(f"Missing ratio: {missing / (missing + present):.4f}")
