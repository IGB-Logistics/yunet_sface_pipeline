#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .sface_model import Sface
from ais_bench.infer.interface import InferSession
import copy

class SfaceOM(Sface):

    def __init__(
        self,
        model_path,
        input_shape=[112,112],
        score_th=0.25,
    ):
        print('--> Load OM model')
        self.session = InferSession(0, model_path)
        super(SfaceOM, self).__init__(input_shape, score_th)

    def __del__(self):
        print('--> Release OM model')
        self.session.free_resource()

    def platform_inference(self, image):
        result = self.session.infer(image, mode='static')
        return result    
    
    # def inference(self, images : list) -> list:
    #     # 前処理
    #     temp_image_list = []
    #     for image in images:
    #         temp_image = copy.deepcopy(image)
    #         temp_image = self._preprocess(temp_image)
    #         temp_image_list.append([temp_image])

    #     # 推論
    #     print('--> Running OM model')
    #     results = self.session.infer_pipeline(temp_image_list, mode='static')

    #     print(results)
    #     return results
