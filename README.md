# CapstoneLPR

### Copy the Repository


```
git clone https://github.com/SotaroKaneda/CapstoneLPR.git  
```
### Install Dependencies  
*Tested with Python>=3.8.0, and PyTorch>=1.8*  
```
cd CapstoneLPR
pip install -r requirements.txt
```
#### Install the appropriate torch and torchvision libraries for your system.  
Install Guide: https://pytorch.org/get-started/locally/  
Commands Below:  
**Linux**  
*CUDA 11.8*  
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
*CUDA 12.1*  
```
pip3 install torch torchvision
```
*CPU*  
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Windows**  
*CUDA 11.8*  
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
*CUDA 12.1*  
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
*CPU*  
```
pip3 install torch torchvision torchaudio
```
**Mac**  
```
pip3 install torch torchvision
```

### Folders and Files  
- main.py : run inference on a an image or a folder of images  
- accuracy-metrics.py : reproduce capstone results or check results from a runs output file produced by main.py  
- scripts/utility.py : support functions 
- scripts/  : Contains several scripts used to label or create datasets 
- classifier/ Resnet50 model, pytorch dataset, and train.py script used for Resnet50 model training  
- extra-code/ : various scripts used for dataset generation and other tasks  
    - *Most of these will need to be modified in some way. None of this code here has been generalized.*
- demo-scripts/ : demo files used throught the year  
- easyocr/ : easyocr prediction scripts  
- trocr/ : trocr prediction scripts  
- yolo : YOLOv5 training README.md

### Run the Model

1. Place the weights folder containing the model weights in the CaptsoneLPR folder.

2. Run main.py with a path to an image or folder of images

```
python main.py "image/or/folder/path"
```
- "image/or/folder/path": path to an image or folder of images
3. Output is stored in model-runs/

### Check Accuracy on Run Output  
**This only works for images from 'data-11-30.csv. The label_file_path variable on line 157 of accuracy_metrics.py will need to be modified for other images not included in the original V2X image set from 11/30.'**
1. Place data-11-30.csv in the CapstoneLPR directory.  
2. Run command:   
```
python accuracy_metrics.py "path/to/runs/output/file"
```
- "path/to/runs/output/file": path to the output file stored in model-runs/ produced by main.py
### Reproduce Capstone Accuracy Metrics  
1. Run command:   
```
python accuracy_metrics.py CAPSTONE "path/to/full" "path/to/small"
```
- "path/to/full": path to 4-25-full-results.csv  
- "path/to/small": path to 4-25-small-results.csv
### Train the model on BigRed

1. Load libraries
   
```
module load python/gpu/3.11.5
pip3 install transformers
pip3 install datasets
pip3 install jiwer
```

2. Run on 1 GPU node (4 GPUS)
```
srun -p gpu --gpus-per-node 1 --nodes 1 -A c00533 python test.py
```
