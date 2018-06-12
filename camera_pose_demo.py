import cv2
import argparse
import chainer
from pose_detector import PoseDetector, draw_person_pose

chainer.using_config('enable_backprop', False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--camera', '-c', type=int, default=2, help='Camera device ID (default is set to 2) check with `v4l2-ctl -d /dev/video0 --list-formats`')
    args = parser.parse_args()

    # load model
    cam_id = args.camera
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)

    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # get video frame
        ret, img = cap.read()

        if not ret:
            print("Failed to capture image")
            break

        person_pose_array, _ = pose_detector(img)
        res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)
        cv2.imshow("result", res_img)
        cv2.waitKey(1)
