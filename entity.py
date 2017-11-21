from enum import IntEnum

from models.CocoPoseNet import CocoPoseNet
from models.FaceNet import FaceNet


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
    },
    'paf_sigma': 1.3,
    'heatmap_sigma': 1.5,
    'crop_iob_thresh': 0.4,
    'crop_size': 480,
    'input_size': 368,
    'downscale': 8,

    'inference_img_size': 368,
    'inference_scales': [0.5, 1.0, 1.5],
    'heatmap_size': 320,
    'gaussian_sigma': 2.5,
    'ksize': 17,
    'n_integ_points': 10,       # 1つのconnectionを10等分して積分計算
    'n_integ_points_thresh': 8, # 1つのconnectionで最低8点以上が閾値を超えた場合に有効
    'heatmap_peak_thresh': 0.1,
    'inner_product_thresh': 0.05,
    'length_penalty_ratio': 0.5,
    'n_subset_limbs_thresh': 7,
    'subset_score_thresh': 0.4,
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
    'face_inference_img_size': 224,
    'face_heatmap_peak_thresh': 0.1,
    'face_crop_scale': 1.5,
}
