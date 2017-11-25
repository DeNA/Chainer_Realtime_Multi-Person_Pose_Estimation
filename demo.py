import cv2
import argparse
from entity import params
import chainer

from models.CocoPoseNet import CocoPoseNet
from models.FaceNet import FaceNet
from models.HandNet import HandNet

from pose_detector import PoseDetector, draw_person_pose
from face_detector import FaceDetector, draw_face_keypoints
from hand_detector import HandDetector, draw_hand_keypoints

chainer.using_config('enable_backprop', False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('--img', help='image file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load model
    pose_detector = PoseDetector("posenet", "models/coco_posenet.npz", device=args.gpu)
    hand_detector = HandDetector("handnet", "models/handnet.npz", device=args.gpu)
    face_detector = FaceDetector("facenet", "models/facenet.npz", device=args.gpu)

    # read image
    img = cv2.imread(args.img)

    # inference
    print("Estimating pose...")
    person_pose_array = pose_detector(img)
    res_img = cv2.addWeighted(img, 0.6, draw_person_pose(img, person_pose_array), 0.4, 0)

    # each person detected
    for person_pose in person_pose_array:
        unit_length = pose_detector.get_unit_length(person_pose)

        # face estimation
        print("Estimating face keypoints...")
        cropped_face_img, bbox = pose_detector.crop_face(img, person_pose, unit_length)
        if cropped_face_img is not None:
            face_keypoints = face_detector(cropped_face_img)
            res_img = draw_face_keypoints(res_img, face_keypoints, (bbox[0], bbox[1]))
            cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

        # hands estimation
        print("Estimating hands keypoints...")
        hands = pose_detector.crop_hands(img, person_pose, unit_length)
        if hands["left"] is not None:
            hand_img = hands["left"]["img"]
            bbox = hands["left"]["bbox"]
            hand_keypoints = hand_detector(hand_img, hand_type="left")
            res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
            cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

        if hands["right"] is not None:
            hand_img = hands["right"]["img"]
            bbox = hands["right"]["bbox"]
            hand_keypoints = hand_detector(hand_img, hand_type="right")
            res_img = draw_hand_keypoints(res_img, hand_keypoints, (bbox[0], bbox[1]))
            cv2.rectangle(res_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 255), 1)

    print('Saving result into result.png...')
    cv2.imwrite('result.png', res_img)
