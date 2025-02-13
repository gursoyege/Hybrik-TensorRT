import tensorrt as trt
import torch
from hybrik.hybrik import Simple3DPoseBaseSMPLCam
model = Simple3DPoseBaseSMPLCam()
img = torch.randn(1, 3, 256, 256)
pose_skeleton, betas, phis, camera_root = model(img)

torch.onnx.export(model, (img), "hybrik.onnx", input_names=["img"], output_names=["pose_skeleton", "betas", "phis", "camera_root"], dynamic_axes={"img": [0]})



def build_model(onnx_file_path):
    engine_file_path = onnx_file_path.replace('.onnx', '.engine')
    build_engine(onnx_file_path, engine_file_path, True)


def build_engine(onnx_file_path, engine_file_path, half=True):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # vitpose
    # profile = builder.create_optimization_profile()
    # profile.set_shape('input', (1,3,256,192), (32,3,256,192), (64,3,256,192)) 
    # config.add_optimization_profile(profile)
    # yolo
    # profile = builder.create_optimization_profile()
    # profile.set_shape('input', (1,3,416,416), (1,3,416,416), (1,3,416,416)) 
    # config.add_optimization_profile(profile)
    # simcc
    # profile = builder.create_optimization_profile()
    # profile.set_shape('input', (1,3,384,288), (1,3,384,288), (1,3,384,288)) 
    # config.add_optimization_profile(profile)

    #reid
    # profile = builder.create_optimization_profile()
    # profile.set_shape('base_images', (1,3,256,128), (32,3,256,128), (64,3,256,128)) 
    # config.add_optimization_profile(profile)
    #hybrik
    profile = builder.create_optimization_profile()
    profile.set_shape('img', (1,3,256,256), (32,3,256,256), (64,3,256,256)) 
    config.add_optimization_profile(profile)




    # config.max_workspace_size = 4 * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_file_path)):
        raise RuntimeError(f'failed to load ONNX file: {onnx_file_path}')
    half &= builder.platform_has_fast_fp16
    if half:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_serialized_network(network, config) as engine, open(engine_file_path, 'wb') as f:
        f.write(engine)
    # with builder.build_engine(network, config) as engine, open(engine_file_path, 'wb') as t:
    #     t.write(engine.serialize())
    return engine_file_path

build_model("hybrik.onnx")
