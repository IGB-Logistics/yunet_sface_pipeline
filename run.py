
#yunet

import cv2
import numpy as np
import time 
import copy
import math
import platform
import argparse

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


def draw_sface(image, input_shape, bboxes, landmarks, face_ids, elapsed_time):
    image_width, image_height = image.shape[1], image.shape[0]
    debug_image = copy.deepcopy(image)
    for bbox, landmark, face_id in zip(bboxes, landmarks,
                                               face_ids):
        x1 = int(image_width * (bbox[0] / input_shape[0]))
        y1 = int(image_height * (bbox[1] / input_shape[1]))
        x2 = int(image_width * (bbox[2] / input_shape[0])) + x1
        y2 = int(image_height * (bbox[3] / input_shape[1])) + y1
        
        # バウンディングボックス
        cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # 顔認証ID
        cv2.putText(
            debug_image,
            'Face ID:' + str(face_id),
            (x1, y1 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
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
    
    # 创建解析步骤
    parser = argparse.ArgumentParser(description='Set platfrom and Input image')
    # 添加参数步骤
    parser.add_argument('--image', '-i', type=str, default='./images/G8_faces.jpg', help='Input image path')
    parser.add_argument('--platform', '-p', type=str, default='onnx', help='Platform name')
    # 解析参数步骤  
    args = parser.parse_args()
    
    yunet = None
    sface = None
    if args.platform == 'onnx':
        print('Use ONNX Runtime')
        from yunet.yunet_onnx import YuNetONNX
        from sface.sface_onnx import SfaceONNX
        yunet_path = "./model/yunet_120x160.onnx"
        sface_path = "./model/sface_112x112.onnx"
        yunet = YuNetONNX(model_path=yunet_path)
        sface = SfaceONNX(model_path=sface_path)
    elif args.platform == 'trt':
        print('Use TRT Model')
        from yunet.yunet_trt import YuNetTRT
        from sface.sface_trt import SfaceTRT
        yunet_path = "./model/yunet_120x160.trt"
        sface_path = "./model/sface_112x112.trt"
        yunet = YuNetTRT(model_path=yunet_path)
        sface = SfaceTRT(model_path=sface_path)
    elif args.platform == 'om':
        print('Use OM Model')
        from yunet.yunet_om import YuNetOM
        from sface.sface_om import SfaceOM
        yunet_path = "./model/yunet_120x160.om"
        sface_path = "./model/sface_112x112.om"
        yunet = YuNetOM(model_path=yunet_path)
        sface = SfaceOM(model_path=sface_path)
    elif args.platform == 'rknn':
        print('Use RKNN Model')
        from yunet.yunet_rknn import YuNetRKNN
        from sface.sface_rknn import SfaceRKNN
        yunet_path = "./model/yunet_120x160_for_rk3588.rknn"
        sface_path = "./model/sface_112x112_for_rk3588.rknn"
        yunet = YuNetRKNN(model_path=yunet_path)
        sface = SfaceRKNN(model_path=sface_path)
    
    #image load
    image_path = args.image
    img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

    #yunet
    start_time = time.time()
    bboxes, landmarks, scores = yunet.inference(img)
    elapsed_time = time.time() - start_time

    detect_image = draw_yunet(
        image = img_raw,
        elapsed_time=elapsed_time,
        score_th=0.8,
        input_shape=[160, 120],
        bboxes=bboxes,
        landmarks=landmarks,
        scores=scores,
    )

    cv2.imwrite('./images/detect_image.jpg', detect_image)
    print('Save results to detect_image.jpg!')

    #sface
    start_time = time.time()
    face_image_list = crop_face_images(
                        image = img,
                        input_shape=[160, 120],
                        bboxes=bboxes,
                        landmarks=landmarks,
                        scores=scores,
                        score_th=0.8,
                        );
    feature_vectors = None
    face_ids = []
    score_th = 0.25

    for i, face_img in enumerate(face_image_list):
        result = sface.inference(face_img)
        # 初回推論時のデータ登録
        if feature_vectors is None:
            feature_vectors = copy.deepcopy(np.array([result]))
        # COS類似度計算
        cos_results = sface._cos_similarity(result, feature_vectors)
        max_index = np.argmax(cos_results)
        max_value = cos_results[max_index]
        if max_value < score_th:
            # スコア閾値以下であれば特徴ベクトルリストに追加
            feature_vectors = np.vstack([
                feature_vectors,
                result,
            ])
            face_ids.append(len(feature_vectors)-1)
        else:
            # スコア閾値以上であれば顔認証のIDを追加
            face_ids.append(max_index)
    
    # results = sface.inference(face_image_list)
    # # 初回推論時のデータ登録
    # if feature_vectors is None:
    #     feature_vectors = copy.deepcopy(np.array(results[0][0][0]))
    # for result in results:
    #     # COS類似度計算
    #     result = np.array(result)
    #     print(result.shape)
    #     cos_results = sface._cos_similarity(result, feature_vectors)
    #     max_index = np.argmax(cos_results)
    #     max_value = cos_results[max_index]

    #     if max_value < score_th:
    #         # スコア閾値以下であれば特徴ベクトルリストに追加
    #         feature_vectors = np.vstack([
    #             feature_vectors,
    #             result,
    #         ])
    #         face_ids.append(len(feature_vectors)-1)
    #     else:
    #         # スコア閾値以上であれば顔認証のIDを追加
    #         face_ids.append(max_index)
            
    print(face_ids)
    elapsed_time = time.time() - start_time

    recognition_image = draw_sface(
                image = img_raw,
                input_shape=[160, 120],
                bboxes=bboxes,
                landmarks=landmarks,
                face_ids = face_ids,
                elapsed_time=elapsed_time,
            )

    cv2.imwrite('./images/recognition_image.jpg', recognition_image)
    print('Save results to recognition_image.jpg!')
