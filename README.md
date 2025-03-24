# Hybrik-TensorRT
## Description
Base code of tensorrt inference for Hybrik. Real-time 3D Human pose estimation from single image with Hybrik, converting to onnx and tensorrt.

## Usage
1. prepare smpl source files from hybrik to model
2. prepare resnet weights from hybrik to pretrained
3. solve tensorrt environment.
```[python]
python convert_hybrik.py
```
the code will generate onnx file, tensorrt engine file and output obj.

python demo_video.py --video-name examples/dance.mp4 --out-dir res_dance --save-pk --save-img --engine-path hybrik.engine

pretrained_models/hybrik_hrnet48_wo3dpw.pth

1. Install the requirements of [HybrIK](https://github.com/jeffffffli/HybrIK)

```
pip install pytorch torchvision -c pytorch 
pip install "numpy<1.4" six terminaltables scipy cython "opencv-python<4.6" matplotlib pycocotools tqdm easydict chumpy pyyaml tb-nightly future ffmpeg-python joblib pycocotools
```
2. Install specific requirements for this repo
```
pip install roma onnx onnxruntime onnxsim pycuda
```

2. Download necessary model files in original HybrIK repository from [Google Drive](https://drive.google.com/file/d/1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV/view) and un-zip them in the `${ROOT}` directory.

3. Download model `.pth` files from the [HybrIK repo](https://github.com/jeffffffli/HybrIK) in `${ROOT}/pretrained_weights`.

4. Solve TensorRT environment to generate `hybrik.engine` and `hybrik.onnx`
```[python]
python convert_hybrik.py
```

Test your code with 

```[python]
python test_hybrik.py
python demo_video.py --video-name examples/dance.mp4 --out-dir res_dance --save-pk --save-img --engine-path hybrik.engine
```
## Results
Test GPU: 4090
FPS: 200+

