import os
import glob


no_lp_lp = os.listdir(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\NO-LP\has-lp")
val_images = os.listdir(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\val\images")
train_images = os.listdir(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\train\images")
test_images = os.listdir(r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\test\images")

for txt_file in no_lp_lp:
    name, ext = os.path.splitext(txt_file)
    if name+".png" in train_images:
        print(name)