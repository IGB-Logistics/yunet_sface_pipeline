
#yunet

import cv2
import numpy as np
import time 
import copy
import math
import platform
import argparse
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

g_bboxes = None
g_landmarks = None
g_feature_vectors = None
g_face_ids = []
g_elapsed_time = None

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

HD_1080P = {"WIDTH": 1920, "HEIGHT": 1080}
HD_720P = {"WIDTH": 1280, "HEIGHT": 720}
HD_360P = {"WIDTH": 640, "HEIGHT": 360}

class Camera:
    def __init__(self, resolution=HD_720P):
        self.resolution = resolution
        self.camera_index = self.find_camera_index()
        self.cap = cv2.VideoCapture(self.camera_index)
        self.configure_camera()
        self.frame = None
        self.running = True
        self.thread_cam = threading.Thread(target=self.update, args=())
        self.thread_cam.start()
        self.prev_time = 0
    

    def configure_camera(self):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution['WIDTH'])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution['HEIGHT'])
    
    def release(self):
        self.running = False
        self.thread_cam.join()
        self.thread_servo.join()
        self.cap.release()
        self.ptz.move_with_pid(0, 0)

    @staticmethod
    def find_camera_index():
        max_index_to_check = 10
        for index in range(max_index_to_check):
            cap = cv2.VideoCapture(index)
            if cap.read()[0]:
                cap.release()
                return index
        raise ValueError("No camera found.")


    def update(self):
        prev_time = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # 计算帧率
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                # 将帧率和分辨率信息绘制到图像上
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Resolution: {self.resolution['WIDTH']}x{self.resolution['HEIGHT']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.frame = frame

    def get_frame(self):
        return self.frame

class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/video_feed':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            while True:
                # 从视频流中读取帧。
                time.sleep(0.01)
                frame = camera.get_frame()
                if frame is not None:
                    recognition_image = draw_sface(
                            image = frame,
                            input_shape=[160, 120],
                            bboxes=g_bboxes,
                            landmarks=g_landmarks,
                            face_ids=g_face_ids,
                            elapsed_time=g_elapsed_time,
                    )
                    ret, buffer = cv2.imencode('.jpg', recognition_image)
                    frame = buffer.tobytes()
                    self.wfile.write(b'--frame\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
        if self.path == '/cam':
            self.send_response(200)
            html = """
                <html>
                    <head>
                        <title>USB Camera Streaming</title>
                    </head>
                    <body>
                        <h1>USB Camera Streaming</h1>
                        <img src="/video_feed">
                    </body>
                </html>
                """
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())



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
    
    try:
        resolution=HD_720P
        camera = Camera(resolution=resolution)
        print("Camera initialized")

        server = HTTPServer(('0.0.0.0', 2233), VideoStreamHandler)
        print("Server started")
        threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
        while True:
            time.sleep(0.001)
            frame = camera.get_frame()
            if frame is not None:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                #yunet
                start_time = time.time()
                bboxes, landmarks, scores = yunet.inference(img)
                #sface
                face_image_list = crop_face_images(
                                    image = img,
                                    input_shape=[160, 120],
                                    bboxes=bboxes,
                                    landmarks=landmarks,
                                    scores=scores,
                                    score_th=0.8,
                                    );
                g_bboxes = bboxes
                g_landmarks = landmarks
                face_ids = []
                score_th = 0.25

                for i, face_img in enumerate(face_image_list):
                    if face_img.shape[0] is 0 or face_img.shape[1] is 0:
                        continue
                    result = sface.inference(face_img)
                    # 初回推論時のデータ登録
                    if g_feature_vectors is None:
                        g_feature_vectors = copy.deepcopy(np.array([result]))
                    # COS類似度計算
                    cos_results = sface._cos_similarity(result, g_feature_vectors)
                    max_index = np.argmax(cos_results)
                    max_value = cos_results[max_index]
                    if max_value < score_th:
                        # スコア閾値以下であれば特徴ベクトルリストに追加
                        g_feature_vectors = np.vstack([
                            g_feature_vectors,
                            result,
                        ])
                        face_ids.append(len(g_feature_vectors)-1)
                    else:
                        # スコア閾値以上であれば顔認証のIDを追加
                        face_ids.append(max_index)
                g_face_ids = face_ids
                g_elapsed_time = time.time() - start_time
                # print(len(g_face_ids), len(g_feature_vectors))

    except KeyboardInterrupt:
        server.shutdown()    # 停止服务器
        camera.release()
        print("Camera released")
        
