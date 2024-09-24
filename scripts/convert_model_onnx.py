import onnx
import onnxscript
import torch
import os


lp_weights = os.path.join(r"C:\Users\Jed\Desktop\capstone_project\CapstoneLPR\best_weights", "v-lp-detect-best.pt")
char_weights = os.path.join(r"C:\Users\Jed\Desktop\capstone_project\CapstoneLPR\best_weights", "x-char-detect-best-2.pt")
lp_model = torch.hub.load('ultralytics/yolov5', 'custom', lp_weights).to("cuda")
char_model = torch.hub.load('ultralytics/yolov5', 'custom', char_weights).to("cuda")
torch_input = torch.randn(1, 3, 640, 640, device="cuda")


torch.onnx.export(lp_model, torch_input, "lp-detect.onnx")
torch.onnx.export(char_model, torch_input, "char-detect.onnx")
