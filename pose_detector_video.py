import os
import cv2
import math
import time
import argparse
import numpy as np
from tqdm import tqdm

import chainer

from entity import params, JointType
from pose_detector import PoseDetector, draw_person_pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('arch', choices=params['archs'].keys(), default='posenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--video', '-v', default=None, help='video file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--result', '-r', default=None, help='result image directory path')
    parser.add_argument('--precise', action='store_true', help='do precise inference')
    args = parser.parse_args()

    chainer.config.enable_backprop = False
    chainer.config.train = False

    # load model
    pose_detector = PoseDetector(args.arch, args.weights, device=args.gpu, precise=args.precise)

    # read video
    cap = cv2.VideoCapture(args.video)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # make result directory
    os.makedirs(args.result, exist_ok=True)
    
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        # inference
        poses, _ = pose_detector(frame)
        # draw and save image
        frame = draw_person_pose(frame, poses)
        # save image
        basename = '{}.png'.format(i)
        image_path = os.path.join(args.result, basename)
        cv2.imwrite(image_path, frame)
    