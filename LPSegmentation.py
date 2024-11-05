from ultralytics import YOLO
import onnx2torch
import torch

# this grabs the pre-trained model and loads -> model
# model = YOLO('LPbest.pt')

# this trains with the dataset given in LPD.yaml, for an epoch of 10, image size of 640, on MPS device (apple processing api)
# model.train(data='LPD.yaml',epochs=10,batch=128,imgsz=640,device='mps')

# model.export(format='engine',device='mps')

torch_model = onnx2torch.convert("modelWeights/Charbest.onnx")
torch.save(torch_model, "modelWeights/Charbest.pt")