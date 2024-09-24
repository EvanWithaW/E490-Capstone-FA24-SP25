import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import os
from PIL import Image
import numpy as np
def annotation_to_box(image, annotation_info):
    
    klass, x_center, y_center, box_width, box_height = annotation_info.split(' ')
    image_height, image_width, channels = np.asarray(image).shape
 
    x_center = float(x_center) * image_width
    y_center = float(y_center) * image_height
    box_width = float(box_width) * image_width
    box_height = float(box_height) * image_height
    return x_center - (.5 * box_width), y_center - (.5 * box_height), box_width, box_height
    # xmin = math.floor(x_center - (box_width/2))
    # ymin = math.floor(y_center - (box_height/2))
    # xmax = math.ceil(x_center + (box_width/2))
    # ymax = math.ceil(y_center + (box_height/2))

    # return [xmin, ymin, xmax, ymax]
def gen_plt(img_paths, titles, img_name):
    fig, axes = plt.subplots(1, len(img_paths), figsize=(15, 5))
    for i, (path, title) in enumerate(zip(img_paths, titles)):
        img = Image.open(path)
        curr_ax = axes[i]
        curr_ax.imshow(img)
        curr_ax.set_title(title)
        curr_ax.axis('off')
        if i == 1:
            curr_ax.set_title(title+ ": annotated")

            with open(f"test_set/lp-predictions/{img_name}.txt", 'r') as f:
                annotation = f.readline()
            x, y, width, height = annotation_to_box(img, annotation)
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
            curr_ax.add_patch(rect)
            with open(f"test_set/lp-annotations/{img_name}.txt", 'r') as f:
                annotation = f.readline()
            x, y, width, height = annotation_to_box(img, annotation)
            rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='g', facecolor='none', alpha = 0.5)
            curr_ax.add_patch(rect)
        if i == 3:
            curr_ax.set_title(title+ ": annotated")

            with open(f"test_set/char-predictions/{img_name}.txt", 'r') as f:
                annotations = f.readlines()
            for annotation in annotations:
                x, y, width, height = annotation_to_box(img, annotation)
                rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
                curr_ax.add_patch(rect)
            with open(f"test_set/char-annotations/{img_name}.txt", 'r') as f:
                annotations = f.readlines()
            for annotation in annotations:
                x, y, width, height = annotation_to_box(img, annotation)
                rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='g', facecolor='none')
                curr_ax.add_patch(rect)
            # box = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')    
        fig.tight_layout()
        fig.savefig(f"{img_name}.png", dpi = 500)

# Example usage:
img_dir = 'test_set'
img_names = ["090008_1693568192990F04_911", "090008_1693568193472R04_581"]

for img_name in img_names:
    directories = [('whole-images', '.jpg'), ('whole-images', '.jpg'), ('lp-cropped-images', '.jpg'), ('lp-cropped-images', '.jpg'),]
    image_paths = [os.path.join(img_dir, directory[0], img_name)+directory[1] for directory in directories]
    char_path = os.path.abspath(os.path.join(img_dir, "char-images", img_name.strip(".")))
    char_img_paths = [os.path.join(char_path, ch_img) for ch_img in os.listdir(char_path)]
    image_paths.extend(char_img_paths)
    image_titles = [x[0] for x in directories]
    # image_titles = ['Title 1', 'Title 2', 'Title 3', 'Title 4']
    print(image_paths)
    image_titles.extend([f"char{x+1}" for x in range(len(char_img_paths))])
    gen_plt(image_paths, image_titles, img_name)
