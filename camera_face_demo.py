import cv2
import argparse
import chainer
import numpy as np
from entity import params
from face_detector import FaceDetector, draw_face_keypoints

chainer.using_config('enable_backprop', False)

def crop_face(img, rect):
    orig_img_h, orig_img_w, _ = img.shape
    crop_center_x = rect[0] + rect[2] / 2
    crop_center_y = rect[1] + rect[3] / 2
    crop_width = rect[2] * params['face_crop_scale']
    crop_height = rect[3] * params['face_crop_scale']
    crop_left = max(0, int(crop_center_x - crop_width / 2))
    crop_top = max(0, int(crop_center_y - crop_height / 2))
    crop_right = min(orig_img_w, int(crop_center_x + crop_width / 2))
    crop_bottom = min(orig_img_h, int(crop_center_y + crop_height / 2))
    cropped_face = img[crop_top:crop_bottom, crop_left:crop_right]
    max_edge_len = np.max(cropped_face.shape[:-1])
    padded_face = np.zeros((max_edge_len, max_edge_len, cropped_face.shape[-1]), dtype=np.uint8)
    padded_face[0:cropped_face.shape[0], 0:cropped_face.shape[1]] = cropped_face
    return padded_face, (crop_left, crop_top)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face detector')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--camera', '-c', type=int, default=2, help='Camera device ID (default is set to 2) check with `v4l2-ctl -d /dev/video0 --list-formats`')
    args = parser.parse_args()

    # load model
    cam_id = args.camera
    face_detector = FaceDetector("facenet", "models/facenet.npz", device=args.gpu)
    cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")

    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # get video frame
        ret, img = cap.read()

        if not ret:
            print("Failed to capture image")
            break

        # crop face
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        facerects = cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3, minSize=(1, 1))
        res_img = img.copy()
        if len(facerects) > 0:
            for facerect in facerects:
                cv2.rectangle(res_img, (facerect[0], facerect[1]), (facerect[0] + facerect[2], facerect[1] + facerect[3]), (255, 255, 255), 1)
                cropped_face, face_left_top = crop_face(img, facerect)
                face_keypoints = face_detector(cropped_face)
                res_img = draw_face_keypoints(res_img, face_keypoints, face_left_top)

        cv2.imshow("result", res_img)
        cv2.waitKey(1)
