# yunet_sface_pipeline
基于人脸yunet检测、sface识别、各硬件平台离线模型的实现。

### 运行脚本：
```python3 run.py -p trt```

-p 选择硬件平台: trt om rknn onnx

运行结果保存在 *./images/detect_image.jpg* 和 *./images/recognition_image.jpg*中。

### 转换脚本
放在 **shll** 文件夹里面，根据情况修改代码，将onnx模型转换成对应硬件的模型。

### 离线模型
放在 **model** 文件夹里面。