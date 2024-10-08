


truth_f = open("../test_set/ground-truth-test-set.csv")
# pred_f = open("easyocr-test-set-cropped.csv")
pred_f = open("../test_set/openalpr_test_set_whole.csv")
data = pred_f.readlines()[1:]
data_truth = truth_f.readlines()[1:]

data_dict = {}
truth_dict = {}
true = 0
false = 0

for entry in data:
    image, read, confidence = entry.split(",")
    data_dict[image.strip()] = {"read":read, "conf":confidence}

for entry in data_truth:
    image, read = entry.split(",")
    
    truth_dict[image.strip()] = {"read":read}

for key, data in truth_dict.items():    
    if truth_dict[key]["read"].strip() == data_dict[key]["read"].strip():
        true += 1
    else:
        false += 1

print(f"True: {true}, False: {false}")
