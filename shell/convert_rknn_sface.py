import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
import copy
import math
from sface_onnx import SfaceONNX
from rknn.api import RKNN

TARGET_PLATFORM = 'rk3588'
QUANTIZE_ON = False
IMG_SIZE = 112

# Model from https://github.com/airockchip/rknn_model_zoo
ONNX_MODEL = 'sface_112x112.onnx'
RKNN_MODEL = 'sface_112x112_for_'+TARGET_PLATFORM+'.rknn'
IMG_PATH = './lena.jpg'
DATASET = './dataset.txt'

LENA_EMB = np.array([
 -1.5394627,   0.34249583,  0.7333064,  -0.05453587,  0.7086928,  -0.6344899,
 -1.2960855,  -1.4240092,  -0.2705814,   1.0025315,  -0.38462642,  0.3207445,
  1.6988405,   1.7204012,  -0.26124895, -0.14901072, -1.791353,   -1.3166926,
 -0.8864413,   0.24839912, -2.6847854,  -0.2620045,   1.0995505,  -1.3623883,
 -1.4629655,  -0.02349555, -0.97532123,  0.29627246, -0.39856237, -0.31119692,
  0.91607594, -0.1974673,   1.0254455,  -1.218931,    0.07291829,  0.2815424,
 -0.40511006,  0.4018502,   0.19861718, -0.97706294,  0.38463315,  1.1459022,
  0.9911314,   0.90373147,  0.6787618,  -1.1776929,  -0.60023844, -0.40719557,
 -0.25173923, -0.426375,   -1.235989,    0.41452298,  0.73230445, -0.74744195,
  0.95784104,  0.3384956,  -0.0406673,  -0.90589505,  0.27176088, -0.39338434,
 -2.283691,    1.3379341,  -1.3977479,  -0.34545875,  0.79826295, -0.79285043,
 -0.11715266, -0.14988573,  0.47345728, -0.34121117, -1.1072971,   0.2966197,
 -1.1811634,  -0.7764927,  -0.81493706, -0.5084669,  -0.62669283,  0.00814208,
 -1.368919,   -1.9557482,  -0.75677824, -0.47746986,  1.0244607,   1.5251638,
 -0.03801738,  0.5288121,   0.22855546,  0.95216036, -0.6347804,   0.6567183,
  1.3144267,   0.686641,    0.63380164,  1.028616,   -0.29133013, -0.6304389,
  0.39302078, -0.94751006, -0.48847675,  1.0448421,  -1.074029,    0.3166448,
  0.32991058, -0.9793569,   0.57801825,  0.3146465,   0.42022893, -0.19657104,
  0.8414223,  -0.5529216,   0.90469086,  0.02954705, -0.0403021,   0.03710273,
 -1.5035315,  -1.7579231,  -0.24564417, -0.45513618,  0.35484686,  0.22890681,
 -1.975252,   -0.29288676,  0.3600944,   0.45343065,  0.28900275, -0.9471799,
 -0.28788036,  0.6358911 ])
    
if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    rknn.config(target_platform=TARGET_PLATFORM)
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    sface = SfaceONNX(model_path=ONNX_MODEL)
    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inference
    print('--> Running model')
    img1 = sface._preprocess(img)
    outputs = rknn.inference(inputs=[img1], data_format=['nchw'])
    print('done')
    
    distance = sface._cos_similarity(np.array(outputs[0][0]), LENA_EMB)
    # post process
    print(distance)

    rknn.release()
