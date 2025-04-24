from ultralytics import YOLO

model = YOLO("Charbest.pt")
model.export(format="engine", device=0)