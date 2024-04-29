#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .yunet_model import YuNet
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

class YuNetTRT(YuNet):

    def __init__(
        self,
        model_path,
        input_shape=[160, 120],
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
    ):
        print('--> Load TRT model')
        # 加载TensorRT引擎
        with open(model_path, 'rb') as f, trt.Runtime(trt.Logger()) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        # 创建执行上下文
        self.context = engine.create_execution_context()
        # 分配内存
        self.inputs = []
        self.outputs = []
        self.bindings = []
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            print(f'Binding: {binding}, Size: {size}, Dtype: {dtype}')
            # 分配内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # 添加到列表
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
        
        super(YuNetTRT, self).__init__(input_shape, conf_th, nms_th, topk, keep_topk)

    def __del__(self):
        print('--> Release TRT model')

    def platform_inference(self, image):
        # 将输入数据复制到GPU
        temp_image = np.ascontiguousarray(image)
        cuda.memcpy_htod(self.inputs[0]['device'], temp_image)
        # 执行模型
        result = self.context.execute(batch_size=1, bindings=self.bindings)
        # 将输出数据从GPU复制到CPU
        for i in range(len(self.outputs)):
            cuda.memcpy_dtoh(self.outputs[i]['host'], self.outputs[i]['device'])
        result = [self.outputs[0]['host'].reshape(-1, 14), 
                  self.outputs[1]['host'].reshape(-1, 2), 
                  self.outputs[2]['host'].reshape(-1, 1) ]
        return result
