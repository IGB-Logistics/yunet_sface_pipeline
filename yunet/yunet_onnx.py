#!/usr/bin/env python
# -*- coding: utf-8 -*-
import onnxruntime
from .yunet_model import YuNet

class YuNetONNX(YuNet):

    def __init__(
        self,
        model_path,
        input_shape=[160, 120],
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
    ):
        print('--> Load ONNX model')
        self.onnx_session = onnxruntime.InferenceSession(model_path)

        self.input_name = self.onnx_session.get_inputs()[0].name
        output_name_01 = self.onnx_session.get_outputs()[0].name
        output_name_02 = self.onnx_session.get_outputs()[1].name
        output_name_03 = self.onnx_session.get_outputs()[2].name
        self.output_names = [output_name_01, output_name_02, output_name_03]
        super(YuNetONNX, self).__init__(input_shape, conf_th, nms_th, topk, keep_topk)

    def __del__(self):
        print('--> Release ONNX model')

    def platform_inference(self, image):
        result = self.onnx_session.run(
            self.output_names,
            {self.input_name: image},
        )
        return result

