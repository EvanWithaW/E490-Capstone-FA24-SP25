from ultralytics import YOLO

# Load model
model = YOLO('yolo11s.pt')

# This trains with the dataset given in LPD.yaml
model.train(
    data='LPSeg.yaml',
    epochs=300,
    batch=16,
    imgsz=1280,
    device=[0,1,2,3],
    workers=4,
    patience=50,
    lr0=0.001,
    lrf=0.01,
    degrees=5.0,
    shear=5.0,
    perspective=0.0005,
    box=0.05,
    cls=0.7,
    overlap_mask=True,
    mask_ratio=8,
    iou=0.4,
    single_cls=False,
    rect=True,
    mosaic=0.8,
    fraction=1.0,
    dropout=0.2,
    mixup=0.1,
    optimizer='AdamW',
    cos_lr=True,
    close_mosaic=10,
    warmup_epochs=5,
    nbs=64,
    cache='disk'
    )

