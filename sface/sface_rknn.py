#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .sface_model import Sface
import platform
from rknnlite.api import RKNNLite

class SfaceRKNN(Sface):

    def __init__(
        self,
        model_path,
        input_shape=[112,112],
        score_th=0.25,
    ):
        host_name = self._get_host()
        if host_name == 'RK3566_RK3568':
            rknn_model = model_path
        elif host_name == 'RK3562':
            rknn_model = model_path
        elif host_name == 'RK3588':
            rknn_model = model_path
        else:
            print("This demo cannot run on the current platform: {}".format(host_name))
            exit(-1)
        # Load RKNN model
        print('--> Load RKNN model')
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(rknn_model)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('done')
        # Init runtime environment
        print('--> Init runtime environment')
        # run on RK356x/RK3588 with Debian OS, do not need specify target.
        if host_name == 'RK3588':
            # For RK3588, specify which NPU core the model runs on through the core_mask parameter.
            ret = self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
        else:
            ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

        super(SfaceRKNN, self).__init__(input_shape, score_th)

    def __del__(self):
        print('--> Release RKNN model')
        self.rknn_lite.release()

    def platform_inference(self, image):
        result = self.rknn_lite.inference(inputs=[image], data_format=['nchw'])
        return result    
    
    def _get_host(self):
        # decice tree for RK356x/RK3588
        DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'
        # get platform and device type
        system = platform.system()
        machine = platform.machine()
        os_machine = system + '-' + machine
        if os_machine == 'Linux-aarch64':
            try:
                with open(DEVICE_COMPATIBLE_NODE) as f:
                    device_compatible_str = f.read()
                    if 'rk3588' in device_compatible_str:
                        host = 'RK3588'
                    elif 'rk3562' in device_compatible_str:
                        host = 'RK3562'
                    else:
                        host = 'RK3566_RK3568'
            except IOError:
                print('Read device node {} failed.'.format(DEVICE_COMPATIBLE_NODE))
                exit(-1)
        else:
            host = os_machine
        return host