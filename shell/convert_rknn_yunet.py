import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2
import copy
import math
from yunet_onnx import YuNetONNX
from rknn.api import RKNN

TARGET_PLATFORM = 'rk3588'
QUANTIZE_ON = False
IMG_SIZE_W = 160
IMG_SIZE_H = 120

# Model from https://github.com/airockchip/rknn_model_zoo
ONNX_MODEL = 'yunet_120x160.onnx'
RKNN_MODEL = 'yunet_120x160_for_'+TARGET_PLATFORM+'.rknn'
IMG_PATH = './G8_faces.jpg'
DATASET = './dataset.txt'

def crop_face_images(image, input_shape, bboxes, landmarks, scores, score_th):
    image_height, image_width = image.shape[0], image.shape[1]

    face_image_list = []
    for score, bbox, landmark in zip(scores, bboxes, landmarks):
        if score_th > score:
            continue
        x1 = int(image_width * (bbox[0] / input_shape[0]))
        y1 = int(image_height * (bbox[1] / input_shape[1]))
        x2 = int(image_width * (bbox[2] / input_shape[0])) + x1
        y2 = int(image_height * (bbox[3] / input_shape[1])) + y1
    
        face_image = copy.deepcopy(image[y1:y2, x1:x2])

        right_eye = landmark[0]
        left_eye = landmark[1]
        mouth = landmark[2]

        a = np.array([((right_eye[0] + left_eye[0]) / 2),
                    ((right_eye[1] + left_eye[1]) / 2)])
        b = np.array([mouth[0], mouth[1]])
        vec = b - a
        angle = math.degrees(np.arctan2(vec[0], vec[1]))

        # face_image = self._image_rotate(face_image, -angle)
        face_image_list.append(face_image)

    return face_image_list
    
def draw_yunet(
    image,
    elapsed_time,
    score_th,
    input_shape,
    bboxes,
    landmarks,
    scores,
):
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)

    for bbox, landmark, score in zip(bboxes, landmarks, scores):
        if score_th > score:
            continue

        # 顔バウンディングボックス
        x1 = int(image_width * (bbox[0] / input_shape[0]))
        y1 = int(image_height * (bbox[1] / input_shape[1]))
        x2 = int(image_width * (bbox[2] / input_shape[0])) + x1
        y2 = int(image_height * (bbox[3] / input_shape[1])) + y1

        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # スコア
        cv2.putText(debug_image, '{:.4f}'.format(score), (x1, y1 + 12),
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0))

        # 顔キーポイント
        for _, landmark_point in enumerate(landmark):
            x = int(image_width * (landmark_point[0] / input_shape[0]))
            y = int(image_height * (landmark_point[1] / input_shape[1]))
            cv2.circle(debug_image, (x, y), 2, (0, 255, 0), 2)

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image
    
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

    yunet = YuNetONNX(
        model_path=ONNX_MODEL,
        input_shape=[IMG_SIZE_W, IMG_SIZE_H],
    )
    # Set inputs
    img = cv2.imread(IMG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Inference
    print('--> Running model')
    img1 = yunet._preprocess(img)
    outputs = rknn.inference(inputs=[img1], data_format=['nchw'])
    print('done')
    # print(outputs)
    # print('----------------------------')

    # post process
    bboxes, landmarks, scores = yunet._postprocess(outputs)
    # bboxes, landmarks, scores = yunet.inference(img)
    print('bboxes:', bboxes)
    detect_image = draw_yunet(
        image = img,
        elapsed_time=0,
        score_th=0.8,
        input_shape=[IMG_SIZE_W, IMG_SIZE_H],
        bboxes=bboxes,
        landmarks=landmarks,
        scores=scores,
    )
    cv2.imwrite('result.jpg', detect_image)
    print('Save results to result.jpg!')

    rknn.release()
