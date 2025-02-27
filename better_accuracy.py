import sys
import PreviousYearWork.scripts.utility as utils
import pandas as pd

results_path = sys.argv[1]
label_path = sys.argv[2]
label_df = pd.read_csv(label_path)

def extract_image_name(view):
    image = view.rsplit('/', 1)[-1]
    return image.split('.jpg')[0]

with open(results_path, "r") as file:
    pred, image, conf = utils.parse_csv(file)

# for every ufm id i need up to 4 predictions and the confidence for each so i can pick the highest
# so i need for each ufm id to search for a corresponding view in the prediction file and append the view, prediciton, and confidence so i can 

ufm_dict = label_df.set_index("UFM_ID")[["PLATE_READ", "IMAGE1", "IMAGE2", "IMAGE3", "IMAGE4"]].to_dict(orient="index")

for key,val in ufm_dict.items():
    for key2, val2 in val.items():
        if(key2.startswith("IMAGE") and type(val2) == str):
            val[key2] = extract_image_name(val2)


print(next(iter(ufm_dict.items())))

for key in ufm_dict