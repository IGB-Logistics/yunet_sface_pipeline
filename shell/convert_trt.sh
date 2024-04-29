trtexec --onnx=../model/yunet_120x160.onnx --saveEngine=../model/yunet_120x160.trt --workspace=1024 --fp16
trtexec --onnx=../model/sface_112x112.onnx --saveEngine=../model/sface_112x112.trt --workspace=1024 --fp16
