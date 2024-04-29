#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .sface_model import Sface
import onnxruntime

class SfaceONNX(Sface):

    def __init__(
        self,
        model_path,
        input_shape=[112,112],
        score_th=0.25,
    ):
        print('--> Load ONNX model')
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.onnx_session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=so)

        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        super(SfaceONNX, self).__init__(input_shape, score_th)
        
    def __del__(self):
        print('--> Release ONNX model')

    def platform_inference(self, image):
        result = self.onnx_session.run(
            None,
            {self.input_name: image},
        )
        return result    
    