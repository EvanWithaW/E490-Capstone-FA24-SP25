from ultralytics import YOLO

model = YOLO('yolo11s.pt')

model.train(data='LPD.yaml',epochs=10,imgsz=640,device='mps')
