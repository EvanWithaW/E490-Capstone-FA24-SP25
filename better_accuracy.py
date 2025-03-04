import sys

import pandas as pd

results_path = sys.argv[1]
label_path = sys.argv[2]
label_df = pd.read_csv(label_path)

results_df = pd.read_csv(results_path)
results_dict = {row["IMAGE"]: (row["PRED"], row["CONF"]) for _, row in results_df.iterrows()}


# print(next(iter(results_dict.items())))

def extract_image_name(view):
    image = view.rsplit('/')[-1]
    return image.split('.jpg')[0]


ufm_dict = label_df.set_index("UFM_ID")[["PLATE_READ", "IMAGE1", "IMAGE2", "IMAGE3", "IMAGE4"]].to_dict(orient="index")

for key, val in ufm_dict.items():
    for key2, val2 in val.items():
        if (key2.startswith("IMAGE") and type(val2) == str):
            val[key2] = extract_image_name(val2)

# for each image in every ufm id row, search the results dict for matching image, if/when found, append (pred, conf) to the new ufm dict. now we have up to 4 pred,conf pairs for every row

sorted_results_dict = {}  # (ufm_id, [(prediction, conf),(prediction, conf),...]) for each available image angle
for ufm_id, vals in ufm_dict.items():  # val is image1: name, image2: name, etc. and plate read
    for key2, val2 in vals.items():
        if key2.startswith("IMAGE"):
            image_id = vals[key2]
            if image_id in results_dict:
                sorted_results_dict.setdefault(ufm_id, []).append(results_dict[image_id])

# print(next(iter(sorted_results_dict.items())))    

filtered_results_dict = {}  # each ufm_id row with only the highest confidence for the (pred,conf) pairs
for ufm_id, tuples in sorted_results_dict.items():
    filtered_results_dict[ufm_id] = max(tuples, key=lambda x: x[1])

tp, fp, tn, fn = 0, 0, 0, 0
manual, auto = 0, 0
missing, present = 0, 0
for ufm_id, vals in ufm_dict.items():
    if ufm_id not in filtered_results_dict:
        missing += 1
        continue
    else:
        present += 1
    lp_label = vals["PLATE_READ"]
    lp_pred = filtered_results_dict[ufm_id][0]
    lp_conf = filtered_results_dict[ufm_id][1]
    if lp_conf < 910:
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

print(f"Accuracy: {(tp + tn) / (tp + tn + fp + fn):.4f}")
print(f"Precision: {tp / (tp + fp):.4f}")
print(f"Recall: {tp / (tp + fn):.4f}")
print(f"F1 Score: {2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))):.4f}")
print(f"Automation Rate: {auto / (auto + manual):.4f}")
print(f"Missing ratio: {missing / (missing + present):.4f}")
