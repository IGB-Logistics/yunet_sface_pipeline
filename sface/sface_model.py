#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import cv2 as cv
import numpy as np
from abc import abstractmethod

class Sface(object):

    def __init__(
        self,
        input_shape=[112,112],
        score_th=0.25,
    ):
        # 各種設定
        self.input_shape = input_shape  # [w, h]
        self.score_th = score_th

    def inference(self, image):
        # 前処理
        temp_image = copy.deepcopy(image)
        temp_image = self._preprocess(temp_image)
        # 推論
        result = self.platform_inference(temp_image)
        result = np.array(result[0][0])

        # 後処理
        # self._postprocess(result)

        return result
    
    @abstractmethod
    def platform_inference(self, image):
        pass

    def _preprocess(self, image):
        assert isinstance(image, np.ndarray), 'image must be numpy array'
        image = cv.resize(
            image,
            (self.input_shape[0], self.input_shape[1]),
            interpolation=cv.INTER_LINEAR,
        )
        image = image.transpose(2, 0, 1)
        image = image.astype('float32')
        image = np.expand_dims(image, axis=0)
        
        return image

    def _postprocess(self, result):
        return result

    
    def _cos_similarity(self, X, Y):
        Y = Y.T
        # (128,) x (n, 128) = (n,)
        result = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y, axis=0))
        return result
    


    def _image_rotate(self, image, angle, scale=1.0):
        image_width, image_height = image.shape[1], image.shape[0]
        center = (int(image_width / 2), int(image_height / 2))

        rotation_mat_2d = cv.getRotationMatrix2D(center, angle, scale)

        result_image = cv.warpAffine(
            image,
            rotation_mat_2d,
            (image_width, image_height),
            flags=cv.INTER_CUBIC,
        )

        return result_image