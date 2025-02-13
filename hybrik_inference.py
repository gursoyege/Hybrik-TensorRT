
import os
from abc import ABCMeta, abstractmethod
from typing import Any

import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pre_processings import bbox_xyxy2cs, top_down_affine
import torch
import roma
from hybrik.layers.smpl.SMPL import SMPL_layer
def check_mps_support():
    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        return 'MPSExecutionProvider' in providers or 'CoreMLExecutionProvider' in providers
    except ImportError:
        return False

class HybrikTensorRT(metaclass=ABCMeta):

    def __init__(self,
                 engine_path: str = None,
                 model_input_size: tuple = None,
                 mean: tuple = None,
                 std: tuple = None,
                 device: str = 'cuda'):

        backend = "tensorrt"
        assert os.path.exists(engine_path)
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()

        self.engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
     

        print(f'load {engine_path} with {backend} backend')

        self.model_input_size = model_input_size
        self.mean = mean
        self.std = std
        self.backend = backend
        self.device = device

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        """Implement the actual function here."""
        raise NotImplementedError

    def inference(self, img):
        # build input to (1, 3, H, W)
        n = len(img)
        img = np.stack(img, axis=0).transpose(0,3,1,2)
        img = np.ascontiguousarray(img, dtype=np.float32)
        input = img


        inputs_host = np.ravel(np.ascontiguousarray(img))
        inputs_device = cuda.mem_alloc(inputs_host.nbytes)

        pose_skeleton_host = cuda.pagelocked_empty(n * 29*3, np.float32)
        pose_skeleton_device = cuda.mem_alloc(pose_skeleton_host.nbytes)
        betas_host = cuda.pagelocked_empty(n*10, np.float32)
        betas_device = cuda.mem_alloc(betas_host.nbytes)
        phis_host = cuda.pagelocked_empty(n*23*2, np.float32)
        phis_device = cuda.mem_alloc(phis_host.nbytes)
        cam_host = cuda.pagelocked_empty(n*3, np.float32)
        cam_device = cuda.mem_alloc(cam_host.nbytes)


        cuda.memcpy_htod_async(inputs_device, inputs_host, self.stream)

        self.context.set_binding_shape(0, input.shape)
        self.context.set_input_shape("img", input.shape)
        self.context.set_tensor_address("img", inputs_device)
        self.context.set_tensor_address("pose_skeleton", pose_skeleton_device)
        self.context.set_tensor_address("betas", betas_device)
        self.context.set_tensor_address("phis", phis_device)
        self.context.set_tensor_address("camera_root", cam_device)

        self.context.execute_async_v3(
            stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(pose_skeleton_host, pose_skeleton_device, self.stream)
        cuda.memcpy_dtoh_async(betas_host, betas_device, self.stream)
        cuda.memcpy_dtoh_async(phis_host, phis_device, self.stream)
        cuda.memcpy_dtoh_async(cam_host, cam_device, self.stream)

        # synchronize stream
        self.stream.synchronize()
        pose_skeleton = np.array(pose_skeleton_host).reshape(n, 29, 3)
        betas = np.array(betas_host).reshape(n, 10)
        phis = np.array(phis_host).reshape(n, 23, 2)
        cam = np.array(cam_host).reshape(n, 3)
        return pose_skeleton, betas, phis, cam

class Hybrik(HybrikTensorRT):

    def __init__(self,
                 engine_path: str,
                 model_input_size: tuple = (256, 256),
                 mean: tuple = (0.406*255, 0.457*255, 0.480*255),
                 std: tuple = (0.225*255, 0.224*255, 0.229*255),
                 device: str = 'cpu'):
        super().__init__(engine_path, model_input_size, mean, std,
                         device)

        h36m_jregressor = np.load('data_info/smpl/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            'data_info/smpl/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl',
            h36m_jregressor=h36m_jregressor,
            dtype=torch.float32
        )

    def __call__(self, image: np.ndarray, bboxes: list = []):
        if len(bboxes) == 0:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        imgs, centers, scales = [], [], []
        for bbox in bboxes:
            img, center, scale = self.preprocess(image, bbox)
            imgs.append(img)
            centers.append(center)
            scales.append(scale)
        pose_skeleton, betas, phis, camera_root = self.inference(imgs) # (N, 17, 64, 48)
        return self.postprocess(pose_skeleton, betas, phis, camera_root, bboxes, centers)
        
    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.)

        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale,
                                             center, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            resized_img = (resized_img - self.mean) / self.std

        return resized_img, center, scale

    def postprocess(
            self, pose_skeleton, betas, phis, camera_root, bboxes, centers
    ):
        output = self.smpl.hybrik(
            pose_skeleton=torch.from_numpy(pose_skeleton).float(),  # unit: meter
            betas=torch.from_numpy(betas).float(),
            phis=torch.from_numpy(phis),
            global_orient=None,
            return_verts=True
        )
        pred_theta_mats = output.rot_mats.float().reshape(-1, 24, 3, 3)
        pred_pose = roma.rotmat_to_rotvec(pred_theta_mats)
        verts = output.vertices

        return betas, pred_pose.cpu().numpy(), verts.cpu().numpy()
        
