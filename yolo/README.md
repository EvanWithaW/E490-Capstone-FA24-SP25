### Setup

1. Download yolo repo and install dependcies
`git clone https://github.com/ultralytics/yolov5  # clone`  
`cd yolov5`  
`pip install -r requirements.txt  # install`  
From: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#2-select-a-model  
 
2. Run `main.py` to run detection on images.  
- This may need to be modified.  
### Train

`python train.py --epochs 100 --data LPD.yaml --weights yolov5s.pt`

- --data : yolo yaml file with dataset path, train folder, validation folder, test folder, and classes. The YOLO Object Detection Tutorial link below shows an example yaml file and dataset folder structure.  
- --weights: pretrained weights  
- Other training parameters can be specified here
https://github.com/ultralytics/yolov5/blob/master/train.py

- Trained weights get put in `\yolov5\runs\train\exp#\weights`  

### Validation  
https://github.com/ultralytics/yolov5/issues/7741


### Links and Tutorials  
Better Training: https://docs.ultralytics.com/yolov5/tutorials/tips_for_best_training_results/  
YOLO Setup: https://medium.com/analytics-vidhya/training-a-custom-object-detection-model-with-yolo-v5-aa9974c07088  
YOLO Custom Training: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data/#local-logging  
YOLO Object Detection Tutorial: https://docs.ultralytics.com/datasets/detect/  
Inference using custom trained model: https://github.com/ultralytics/yolov5/issues/7044  
