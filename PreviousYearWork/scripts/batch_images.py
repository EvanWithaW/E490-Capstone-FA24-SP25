import os
import shutil
import glob

"""
    Separate Images into batches
"""

image_files = glob.glob(os.path.join(r"D:\v2x-11-30-data\crops\images", "*.*"))
num_images = len(image_files)
directory = r"D:\v2x-11-30-data\crops\batches"
foldername = ""
batch_size = 500

# Create batch folders
dir_names = []
batches = []

for x in range(0, num_images, batch_size):
    images = image_files[x:x+batch_size]

    if (x+batch_size) > num_images:
        foldername = f"{directory}/new-batch{x}-{x + (num_images - x)}"
        batches.append(image_files[x:(x + (num_images - x))])
    else:
        foldername = f"{directory}/new-batch{x}-{x+batch_size}"
        batches.append(image_files[x:x+batch_size])

    dir_names.append(foldername)

    if not os.path.exists(foldername):
        os.mkdir(foldername)


for batch, folder in zip(batches, dir_names):
    for file in batch:
        filename = file.split("\\")[-1]
        shutil.copy2(file, os.path.join(folder, filename))