import os
import shutil


images_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\val\images"
labels_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\val\labels"
save_image_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\he-invhe-val\images"
save_label_folder = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\he-invhe-val\labels"
images = os.listdir(images_folder)

# reg = [image for image in images if "INV" not in image and "HE" not in image]
he = [image for image in images if "HE" in image]
# inv_he = [image for image in images if "INV-HE" in image]
for image in he:
    name, ext = os.path.splitext(image)
    txt = name+".txt"
    if not os.path.exists(os.path.join(labels_folder, txt)):
        continue
    shutil.copy2(os.path.join(images_folder, image), os.path.join(save_image_folder, image))
    shutil.copy2(os.path.join(labels_folder, txt), os.path.join(save_label_folder, txt))


