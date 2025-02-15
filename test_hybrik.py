import cv2 as cv
import numpy as np
from hybrik_inference import Hybrik, HybrikOnnx
import time
import torch

pose_model = Hybrik("./hybrik.engine", model_input_size=(256,256), device="cuda")
#pose_model = HybrikOnnx("./hybrik.onnx", model_input_size=(256,256), device="cuda")

img = cv.imread("human-pose.jpg")
dets = np.array([[  0.75959396,   1.6419703,  212.80466,    339.2592    ]])
shape, pose = pose_model(img, dets)
# from vis_tools import Visualizer, PySMPL
# smpl = PySMPL()
# vis = Visualizer()
# shape, pose = pose_model(img, dets)
# mesh = smpl(torch.from_numpy(shape), torch.from_numpy(pose))
# vis.show_points([mesh])
# exit()


for i in range(1000):
    start_time = time.perf_counter()
    
    shape, pose = pose_model(img, dets + np.random.randint(-2,2))
    end_time = time.perf_counter()
    print(torch.isnan(torch.from_numpy(pose)).any())

    print("Time: " + str((end_time - start_time)))
