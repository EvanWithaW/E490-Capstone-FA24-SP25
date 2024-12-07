import cv2
import numpy as np
from pathlib import Path


def process_image(image_path, output_folder):
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    # Add your image processing code here
    output_image = process(img)
    # Save processed image
    output_path = output_folder / image_path.name
    cv2.imwrite(str(output_path), output_image)

def process_folder(input_folder, output_folder):
    # Convert paths to Path objects
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    # Ensure the input folder exists
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
    # Process each image in the folder
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in image_extensions:
            print(f"Processing: {file_path.name}")
            try:
                process_image(file_path, output_path)
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")

def process(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(1,5))
    clahe_output = clahe.apply(gray)
    # blur
    psf = np.ones((5, 5)) / 25
    deblurred = None
    try:
        deblurred = cv2.deconvolution(clahe_output, psf)[1]
    except:
        gaussian = cv2.GaussianBlur(clahe_output, (0, 0), 3)
        deblurred = cv2.addWeighted(clahe_output, 2.0, gaussian, -0.5, 0)
    # Normalize deblurred image
    deblurred = cv2.normalize(deblurred, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return deblurred

if __name__ == "__main__":
    # Specify folder paths
    input_folder = "sample_cropped_images/crops/license plate"
    output_folder = "sample_cropped_images/enhanced_plates"
    process_folder(input_folder, output_folder)