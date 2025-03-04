# This project is a Work in Progress.

## The team is working on a automated license plate recognition (ALPR) system utilizing machine learning.

Our current design will include:

- Efficient and fast machine learning models which can read license plate strings
- Confidence gating system to flag possible wrong predictions
- Generalized scripts and in-depth user manuals on how to use the system

## Project Structure

- `PreviousYearWork/` - Contains all unnecessary files and scripts from previous Capstone groups
- `modelWeights/` - Location to put all weights for the models (LPBest - __LP Localization__ and Charbest - __Character
  Classification and Segmentation__ ) - you ***will*** need to create this folder due to privacy/security.
- `model-runs/` - Automatically generated folder to store model runs (predictions) from model. Each file in this folder
  is a .csv and stores all run predictions
- `All external files` - All files not in the previous three directories are pertinent to this year's project and holds
  scripts or utility functions to help with model inference.

## How to make Predictions

1. Clone the repository
2. Install the required packages using `pip install -r requirements.txt`
3. Replace the LP and Char weights at top of `main.py` with the weights you want to use:
    - `LPBest` - __LP Localization__ model weights
    - `Charbest` - __Character Classification and Segmentation__ model weights
4. Run `main.py` with argument to image folder you want to predict.

Example of main.py run: `python3 main.py "../../../CapstoneDataset"`

- Where `CapstoneDataset/` is the directories holding all license plate images.
- All images must be within `CapstoneDataset/` and cannot be in nested folders.

## How to calculate accuracy of model

1. Clone the repository
2. Install the required packages using `pip install -r requirements.txt`
3. Run a prediction using the steps discussed above in **How to make Predictions**
4. Run `accuracy_metrics.py` with additional argument pointing to the model run csv file from your prediction in step 3.
5. Input the path to the label file for the images you ran predictions for. This can also be supplied in the code if you
   don't want to type it out everytime, its at the bottom of the file.

Example of accuracy_metrics.py run:
`python3 accuracy_metrics.py /Users/evanw/Desktop/GitHubRepos/Capstone/E490-Capstone-FA24-SP25/model-runs/Run---2025-02-10---19-35-29.csv "`