from enum import IntEnum

from models.CocoPoseNet import CocoPoseNet

from models.FaceNet import FaceNet
from models.HandNet import HandNet


class JointType(IntEnum):
    """関節の種類を表す """
    Nose = 0
    """ 鼻 """
    Neck = 1
    """ 首 """
    RightShoulder = 2
    """ 右肩 """
    RightElbow = 3
    """ 右肘 """
    RightHand = 4
    """ 右手 """
    LeftShoulder = 5
    """ 左肩 """
    LeftElbow = 6
    """ 左肘 """
    LeftHand = 7
    """ 左手 """
    RightWaist = 8
    """ 右腰 """
    RightKnee = 9
    """ 右膝 """
    RightFoot = 10
    """ 右足 """
    LeftWaist = 11
    """ 左腰 """
    LeftKnee = 12
    """ 左膝 """
    LeftFoot = 13
    """ 左足 """
    RightEye = 14
    """ 右目 """
    LeftEye = 15
    """ 左目 """
    RightEar = 16
    """ 右耳 """
    LeftEar = 17
    """ 左耳 """

params = {
    'coco_dir': 'coco',
    'archs': {
        'posenet': CocoPoseNet,
        'facenet': FaceNet,
        'handnet': HandNet,
    },
    # training params
    'min_keypoints': 5,
    'min_area': 32 * 32,
    'insize': 368,
    'downscale': 8,
    'paf_sigma': 8,
    'heatmap_sigma': 7,

    'min_box_size': 64,
    'max_box_size': 512,
    'min_scale': 0.5,
    'max_scale': 2.0,
    'max_rotate_degree': 40,
    'center_perterb_max': 40,

    # inference params
    'inference_img_size': 368,
    'inference_scales': [0.5, 1, 1.5, 2],
    # 'inference_scales': [1.0],
    'heatmap_size': 320,
    'gaussian_sigma': 2.5,
    'ksize': 17,
    'n_integ_points': 10,
    'n_integ_points_thresh': 8,
    'heatmap_peak_thresh': 0.05,
    'inner_product_thresh': 0.05,
    'limb_length_ratio': 1.0,
    'length_penalty_value': 1,
    'n_subset_limbs_thresh': 3,
    'subset_score_thresh': 0.2,
    'limbs_point': [
        [JointType.Neck, JointType.RightWaist],
        [JointType.RightWaist, JointType.RightKnee],
        [JointType.RightKnee, JointType.RightFoot],
        [JointType.Neck, JointType.LeftWaist],
        [JointType.LeftWaist, JointType.LeftKnee],
        [JointType.LeftKnee, JointType.LeftFoot],
        [JointType.Neck, JointType.RightShoulder],
        [JointType.RightShoulder, JointType.RightElbow],
        [JointType.RightElbow, JointType.RightHand],
        [JointType.RightShoulder, JointType.RightEar],
        [JointType.Neck, JointType.LeftShoulder],
        [JointType.LeftShoulder, JointType.LeftElbow],
        [JointType.LeftElbow, JointType.LeftHand],
        [JointType.LeftShoulder, JointType.LeftEar],
        [JointType.Neck, JointType.Nose],
        [JointType.Nose, JointType.RightEye],
        [JointType.Nose, JointType.LeftEye],
        [JointType.RightEye, JointType.RightEar],
        [JointType.LeftEye, JointType.LeftEar]
    ],
    'coco_joint_indices': [
        JointType.Nose,
        JointType.LeftEye,
        JointType.RightEye,
        JointType.LeftEar,
        JointType.RightEar,
        JointType.LeftShoulder,
        JointType.RightShoulder,
        JointType.LeftElbow,
        JointType.RightElbow,
        JointType.LeftHand,
        JointType.RightHand,
        JointType.LeftWaist,
        JointType.RightWaist,
        JointType.LeftKnee,
        JointType.RightKnee,
        JointType.LeftFoot,
        JointType.RightFoot
    ],

    # face params
    'face_inference_img_size': 368,
    'face_heatmap_peak_thresh': 0.1,
    'face_crop_scale': 1.5,
    'face_line_indices': [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], # 輪郭
        [17, 18], [18, 19], [19, 20], [20, 21], # 右眉
        [22, 23], [23, 24], [24, 25], [25, 26], # 左眉
        [27, 28], [28, 29], [29, 30], # 鼻
        [31, 32], [32, 33], [33, 34], [34, 35], # 鼻下の横線
        [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36], # 右目
        [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42], # 左目
        [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54], [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48], # 唇外輪
        [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60] # 唇内輪
    ],

    # hand params
    'hand_inference_img_size': 368,
    'hand_heatmap_peak_thresh': 0.1,
    'fingers_indices': [
        [[0, 1], [1, 2], [2, 3], [3, 4]],
        [[0, 5], [5, 6], [6, 7], [7, 8]],
        [[0, 9], [9, 10], [10, 11], [11, 12]],
        [[0, 13], [13, 14], [14, 15], [15, 16]],
        [[0, 17], [17, 18], [18, 19], [19, 20]],
    ],
}
