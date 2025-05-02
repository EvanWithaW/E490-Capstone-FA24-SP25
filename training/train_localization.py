from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt") 

# Train the model with LPD.yaml
results = model.train(data="LPLoc.yaml", epochs=100, batch=64, imgsz=640, patience = 20)
