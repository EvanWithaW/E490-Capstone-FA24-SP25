# Automated License Plate Recognition System
## About

This project is an **Automated License Plate Recognition (ALPR) system** designed as a capstone project for E490. The system uses computer vision and machine learning to identify and extract license plate numbers from images. It processes a batch of image files, makes predictions using a trained model, and exports the results to a CSV file for further analysis. Additionally, accuracy metrics can be generated for performance evaluation.

### Key Features
- Batch image processing for license plate recognition
- Outputs predictions to a structured CSV format
- Accuracy metrics for evaluating model performance
- Modular and extensible codebase for future enhancements



## Getting Started
### Setup
Note: in order to run this project, you will need independently generated model weights. We name the weights we use in the following way:
- `LPBest` - __LP Localization__ model weights
- `Charbest` - __Character Classification and Segmentation__ model weights
These weights are to be stored in the "modelWeights" folder of the repository's home directory. 

In addition to model weights, this project requires a folder of image files and, if wanting to evaluate the accuracy of the model, a label file in .csv format. 
The paths to these files will be specified in the command line when generating a model run and evaluating the performance of the model. 

Once you have obtained the relevant files, you may clone the repository using this command:
```bash
git clone https://github.com/EvanWithaW/E490-Capstone-FA24-SP25
```

Then, create a virtual environment to install the project dependencies:
```bash
python3 -m venv .venv  
```
Activate the environment once it has been created:
```bash
source .venv/bin/activate
```

### Generating a model run

Once you are inside the virtual environment, run this command from the home directory of the repository to install the project dependencies:
```bash
pip install -r requirements.txt
```

After all dependencies are installed and the modelWeights are placed in their corresponding folder, you may run the model and generate predictions using the main.py file:
```bash
python3 main.py /path/to/images/folder/
```

The result will be stored in the model-runs folder.

### Evaluating model results

Once a run file has been generated using the main.py script, you can evaluate the performance of the model using the accuracy.py file.
Running this file requires 3 command line arguments:
- /path/to/label/file
- model-runs/Run-file.csv
- confidence-value

The confidence threshold is intentionally configurable for maximal flexibility, but we recommend starting with a value of 925. This value corresponds to a percentage (925 -> 92.5%) representing how confident the model is about its prediction. Any prediction generated with a confidence value below the confidence threshold is marked to be reviewed manually, and all other predictions are marked as to be predicted automatically. 

To run the accuracy.py file, follow this format:
```bash
python3 accuracy.py //path/to/label/file model-runs/Run-file confidence-value
```

The result will be displayed on the command line. This file measures the following metrics:
- Precision: the percentage of images marked for automatic review that were predicted correctly
- Automation Rate: the percentage of images that the model was confident in predicting
- Accuracy: the percentage of the sum of correctly predicted automated images and incorrectly predicted manual images divided by every image in the dataset
- Recall: the percentage of images that would have been correctly predicted by the model divided by every image in the dataset
- F1 Score: the harmonic mean between precision and recall
- Automatic Plates Read
- Manual Plates Read
- Total Images Read


### Troubleshooting Guide
| Issue                              | Solution                                                                 |
|------------------------------------|--------------------------------------------------------------------------|
| ModuleNotFoundError                | Run `pip install -r requirements.txt`.                                   |
| Missing model weights              | Ensure `LPBest`/`Charbest` are in `modelWeights/`.                       |
| Path errors                        | Use absolute paths (e.g., `/home/user/images/`, not `~/images/`).        |
| Low accuracy                       | Verify label file formatting or adjust confidence threshold.             |
| Division by zero error in `accuracy.py` | Ensure the label file matches the images used in the model run.     |
