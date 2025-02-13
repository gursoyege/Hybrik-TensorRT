import cv2 as cv
import numpy as np
from hybrik_inference import Hybrik
import time
import trimesh
pose_model = Hybrik("./hybrik.engine", model_input_size=(256,256), device="cuda")

img = cv.imread("human-pose.jpg")
dets = np.array([[  0.75959396,   1.6419703,  212.80466,    339.2592    ]])
shape, pose, verts = pose_model(img, dets)

mesh = trimesh.Trimesh(verts.reshape(-1,3), pose_model.smpl.faces)
mesh.export("output.obj")
# for i in range(1000):
#     start_time = time.perf_counter()
#     shape, pose, transl = pose_model(img, dets)
#     end_time = time.perf_counter()
#     print("Time: " + str((end_time - start_time)))
