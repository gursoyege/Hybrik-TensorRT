# Hybrik-TensorRT
## Description
Base code of tensorrt inference for Hybrik. Real-time 3D Human pose estimation from single image with Hybrik, converting to onnx and tensorrt.
![output](vis.png)

## Usage
1. prepare smpl source files from hybrik to data_info
2. prepare resnet weights from hybrik to pretrained
3. solve tensorrt environment.
```[python]
python convert_hybrik.py
python test_hybrik.py
```
the code will generate onnx file, tensorrt engine file and output obj.

## Reference
Hybrik: [https://github.com/jeffffffli/HybrIK](https://github.com/jeffffffli/HybrIK)

