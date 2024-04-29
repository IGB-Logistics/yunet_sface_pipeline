#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .yunet_model import YuNet
from ais_bench.infer.interface import InferSession

class YuNetOM(YuNet):

    def __init__(
        self,
        model_path,
        input_shape=[160, 120],
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
    ):
        print('--> Load OM model')
        self.session = InferSession(0, model_path)
        super(YuNetOM, self).__init__(input_shape, conf_th, nms_th, topk, keep_topk)

    def __del__(self):
        print('--> Release OM model')
        self.session.free_resource()

    def platform_inference(self, image):
        result = self.session.infer([image], mode='static')
        return result
