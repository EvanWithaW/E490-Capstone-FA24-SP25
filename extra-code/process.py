import cv2
import os
import shutil



def HE(img):
    # Convert the image to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)

        # Apply histogram equalization to the L channel
        l_equalized = cv2.equalizeHist(l)

        # Merge the equalized L channel with the original A and B channels
        lab_equalized = cv2.merge((l_equalized, a, b))

        # Convert the equalized LAB image back to BGR color space
        equalized_img = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

        return equalized_img

# Create the output directory if it doesn't exist
output_dir = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\test-he-inv"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count = 0

# Iterate through the images in the input directory
input_dir = r"D:\v2x-11-30-data\11-30-Parsed\TRAIN-TEST\TRAIN-CHAR\test\images"
for filename in os.listdir(input_dir):
    name, ext = os.path.splitext(filename)
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # Read the image
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        inverted = ~img
        equalized_img = HE(inverted)
        # Save the processed image
        path = os.path.join(output_dir, filename)

        cv2.imwrite(path, equalized_img)


        # print(f"Processed image saved: {output_path}")
print("All images processed and saved.")