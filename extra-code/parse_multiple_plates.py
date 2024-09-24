import os
import glob
import shutil


folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\MULTIPLE-PLATES-WHOLE\obj_train_data"
txt_paths = glob.glob(os.path.join(folder, "*.txt"))
move_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\MULTIPLE-PLATES-WHOLE\multiple-preds"
multiple = 0
for path in txt_paths:
    move = False
    with open(path, "r") as file:
        if len(file.readlines()) > 1:
            multiple += 1
            move = True

    if move:    
        base, file = os.path.split(path)
        name, ext = os.path.splitext(file)
        if os.path.exists(path):
            shutil.move(path, os.path.join(move_folder, file))
        shutil.move(os.path.join(folder, name+".jpg"), os.path.join(move_folder, name+".jpg"))

print(multiple)