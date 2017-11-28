import os
import sys
import cv2
import random
import numpy as np
from chainer.dataset import DatasetMixin
from pycocotools.coco import COCO

from entity import JointType, params


class CocoDataLoader(DatasetMixin):
    def __init__(self, coco, mode='train', n_samples=None):
        self.coco = coco
        assert mode in ['train', 'val', 'eval'], 'Data loading mode is invalid.'
        self.mode = mode
        self.catIds = coco.getCatIds(catNms=['person'])
        self.imgIds = sorted(coco.getImgIds(catIds=self.catIds))
        if self.mode in ['val', 'eval'] and n_samples is not None:
            random.seed(2)
            self.imgIds = random.sample(self.imgIds, n_samples)
        print('{} images: {}'.format(mode, len(self)))

    def __len__(self):
        return len(self.imgIds)

    # return shape: (height, width)
    def gen_gaussian_heatmap(self, imshape, joint, sigma):
        x, y = joint
        grid_x = np.tile(np.arange(imshape[1]), (imshape[0], 1))
        grid_y = np.tile(np.arange(imshape[0]), (imshape[1], 1)).transpose()
        grid_distance = (grid_x - x) ** 2 + (grid_y - y) ** 2
        gaussian_heatmap = np.exp(-grid_distance / sigma ** 2)
        return gaussian_heatmap

    # return shape: (2, height, width)
    def gen_constant_paf(self, imshape, joint_from, joint_to, paf_width):
        if np.array_equal(joint_from, joint_to): # same joint
            return np.zeros((2,) + imshape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        vertical_unit_vector = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(imshape[1]), (imshape[0], 1))
        grid_y = np.tile(np.arange(imshape[0]), (imshape[1], 1)).transpose()
        horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)
        vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] * (grid_y - joint_from[1])
        vertical_paf_flag = np.abs(vertical_inner_product) <= paf_width
        paf_flag = horizontal_paf_flag & vertical_paf_flag
        constant_paf = np.stack((paf_flag, paf_flag)) * np.broadcast_to(unit_vector, imshape[:-1] + (2,)).transpose(2, 0, 1)
        return constant_paf

    # return shape: (2, height, width)
    def gen_gaussian_paf(self, imshape, joint_from, joint_to, paf_sigma):
        if np.array_equal(joint_from, joint_to): # same joint
            return np.zeros((2,) + imshape[:-1])

        joint_distance = np.linalg.norm(joint_to - joint_from)
        unit_vector = (joint_to - joint_from) / joint_distance
        rad = np.pi / 2
        rot_matrix = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
        vertical_unit_vector = np.dot(rot_matrix, unit_vector)
        grid_x = np.tile(np.arange(imshape[1]), (imshape[0], 1))
        grid_y = np.tile(np.arange(imshape[0]), (imshape[1], 1)).transpose()
        horizontal_inner_product = unit_vector[0] * (grid_x - joint_from[0]) + unit_vector[1] * (grid_y - joint_from[1])
        horizontal_paf_flag = (0 <= horizontal_inner_product) & (horizontal_inner_product <= joint_distance)
        vertical_inner_product = vertical_unit_vector[0] * (grid_x - joint_from[0]) + vertical_unit_vector[1] * (grid_y - joint_from[1])
        gauss = np.exp(-np.abs(vertical_inner_product) ** 2 / paf_sigma ** 2)
        gaussian_paf = np.stack((horizontal_paf_flag * gauss, horizontal_paf_flag * gauss)) * np.broadcast_to(unit_vector, imshape[:-1] + (2,)).transpose(2, 0, 1)
        return gaussian_paf

    # return shape: (img_height, img_width)
    def gen_ignore_mask(self, ignore_mask, ignore_region):
        ignore_vertices = []
        for vertex in ignore_region:
            ignore_vertices.append([vertex['x'], vertex['y']])
        cv2.fillPoly(ignore_mask, np.array([ignore_vertices]), 255)
        return ignore_mask

    # return shape: (img_height, img_width)
    def gen_ignore_masks(self, ignore_regions, img):
        ignore_mask = np.zeros(img.shape[:-1])
        for ignore_region in ignore_regions:
            ignore_mask = self.gen_ignore_mask(ignore_mask, ignore_region)
        return ignore_mask

    def overlay_paf(self, img, paf):
        hue = ((np.arctan2(paf[1], paf[0]) / np.pi) / -2 + 0.5)
        saturation = np.sqrt(paf[0] ** 2 + paf[1] ** 2)
        saturation[saturation > 1.0] = 1.0
        value = saturation.copy()
        hsv_paf = np.vstack((hue[np.newaxis], saturation[np.newaxis], value[np.newaxis])).transpose(1, 2, 0)
        rgb_paf = cv2.cvtColor((hsv_paf * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        img = cv2.addWeighted(img, 0.7, rgb_paf, 0.3, 0)
        return img

    def overlay_pafs_gaussian(self, img, pafs):
        mix_paf = np.zeros((2,) + img.shape[:-1])
        max_paf_length = np.zeros(img.shape[:-1]) # for gaussian paf

        for paf in pafs.reshape((int(pafs.shape[0]/2), 2,) + pafs.shape[1:]):
            paf_length = np.sqrt(paf[0] ** 2 + paf[1] ** 2)
            max_paf_length = np.maximum(max_paf_length, paf_length)
            mix_paf += paf

        paf_length = np.sqrt(mix_paf[0] ** 2 + mix_paf[1] ** 2)
        paf_length_scale = np.ones(paf_length.shape)
        paf_length_scale[paf_length > 0] = max_paf_length[paf_length > 0] / paf_length[paf_length > 0]
        mix_paf *= np.broadcast_to(paf_length_scale, mix_paf.shape)

        img = self.overlay_paf(img, mix_paf)
        return img

    def overlay_pafs(self, img, pafs):
        mix_paf = np.zeros((2,) + img.shape[:-1])
        paf_flags = np.zeros(mix_paf.shape) # for constant paf

        for paf in pafs.reshape((int(pafs.shape[0]/2), 2,) + pafs.shape[1:]):
            paf_flags = paf != 0
            paf_flags += np.broadcast_to(paf_flags[0] | paf_flags[1], paf.shape)
            mix_paf += paf

        mix_paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
        img = self.overlay_paf(img, mix_paf)
        return img

    def overlay_heatmap(self, img, heatmap):
        rgb_heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        img = cv2.addWeighted(img, 0.7, rgb_heatmap, 0.3, 0)
        return img

    def overlay_ignore_mask(self, img, ignore_mask):
        img = img * np.broadcast_to((ignore_mask == 0).astype(np.uint8), (3,) + ignore_mask.shape).transpose(1, 2, 0)
        return img

    def compute_intersection(self, box1, box2):
        intersection_width =  np.minimum(box1[1][0], box2[1][0]) - np.maximum(box1[0][0], box2[0][0])
        intersection_height = np.minimum(box1[1][1], box2[1][1]) - np.maximum(box1[0][1], box2[0][1])

        if (intersection_height < 0) or (intersection_width < 0):
            return 0
        else:
            return intersection_width * intersection_height

    def compute_area(self, box):
        [[left, top], [right, bottom]] = box
        return (right - left) * (bottom - top)

    # intersection of box
    def compute_iob(self, box, joint_bbox):
        intersection = self.compute_intersection(box, joint_bbox)
        area = self.compute_area(joint_bbox)
        if area == 0:
            iob = 0
        else:
            iob = intersection / area
        return iob

    def validate_crop_area(self, crop_bbox, joint_bboxes, iob_thresh):
        valid_iob_list = []
        iob_list = []
        for joint_bbox in joint_bboxes:
            iob = self.compute_iob(crop_bbox, joint_bbox)
            valid_iob_list.append(iob <= 0 or iob >= iob_thresh)
            iob_list.append(iob)
        return valid_iob_list, np.array(iob_list)

    def random_crop_img(self, orig_img, mask_img, joints, valid_joints, joint_bboxes, min_crop_size):
        # get correct crop area
        iteration = 0
        while True:
            iteration += 1
            crop_width = crop_height = np.random.randint(min_crop_size, min(orig_img.shape[:-1]) + 1)
            crop_left = np.random.randint(orig_img.shape[1] - crop_width + 1)
            crop_top = np.random.randint(orig_img.shape[0] - crop_height + 1)
            crop_right = crop_left + crop_width
            crop_bottom = crop_top + crop_height

            valid_iob_list, iob_list = self.validate_crop_area([[crop_left, crop_top], [crop_right, crop_bottom]], joint_bboxes, params['crop_iob_thresh'])
            if (sum(valid_iob_list) == len(valid_iob_list)) or iteration > 10:
                break

        cropped_img = orig_img[crop_top:crop_bottom, crop_left:crop_right]
        mask_img = mask_img[crop_top:crop_bottom, crop_left:crop_right]
        joints[:, :, 0][valid_joints] -= crop_left
        joints[:, :, 1][valid_joints] -= crop_top
        valid_joints[iob_list == 0, :] = False
        return cropped_img, mask_img, joints, valid_joints

    # distort image color
    def distort_color(self, img):
        img_max = np.broadcast_to(np.array(255, dtype=np.uint8), img.shape[:-1])
        img_min = np.zeros(img.shape[:-1], dtype=np.uint8)

        hsv_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv_img[:, :, 0] = np.maximum(np.minimum(hsv_img[:, :, 0] - 10 + np.random.randint(20 + 1), img_max), img_min) # hue
        hsv_img[:, :, 1] = np.maximum(np.minimum(hsv_img[:, :, 1] - 40 + np.random.randint(80 + 1), img_max), img_min) # saturation
        hsv_img[:, :, 2] = np.maximum(np.minimum(hsv_img[:, :, 2] - 30 + np.random.randint(60 + 1), img_max), img_min) # value
        hsv_img = hsv_img.astype(np.uint8)

        distorted_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

        return distorted_img

    def resize_data(self, img, mask, joints, resize_shape):
        """resize img and mask"""
        img_h, img_w, _ = img.shape

        resized_img = cv2.resize(img, resize_shape)
        resized_mask = cv2.resize(mask.astype(np.uint8), resize_shape).astype('bool')
        joints = (joints * np.array(resize_shape) / np.array((img_w, img_h))).astype(np.int64)

        return resized_img, resized_mask, joints

    def flip_img(self, img, mask, joints, valid_joints):
        """flip img"""
        flipped_img = cv2.flip(img, 1)
        flipped_mask = cv2.flip(mask.astype(np.uint8), 1).astype('bool')
        center_pos = (np.array(img.shape[:-1]) / 2).astype(np.int64)
        joints[:, :, 0] = center_pos[0] + center_pos[0] - joints[:, :, 0]

        def swap_joints(joints, valid_joints, joint_type_1, joint_type_2):
            tmp = joints[:, joint_type_1, :].copy()
            joints[:, joint_type_1, :] = joints[:, joint_type_2, :]
            joints[:, joint_type_2, :] = tmp

            tmp = valid_joints[:, joint_type_1].copy()
            valid_joints[:, joint_type_1] = valid_joints[:, joint_type_2]
            valid_joints[:, joint_type_2] = tmp

        swap_joints(joints, valid_joints, JointType.LeftEye, JointType.RightEye)
        swap_joints(joints, valid_joints, JointType.LeftEar, JointType.RightEar)
        swap_joints(joints, valid_joints, JointType.LeftShoulder, JointType.RightShoulder)
        swap_joints(joints, valid_joints, JointType.LeftElbow, JointType.RightElbow)
        swap_joints(joints, valid_joints, JointType.LeftHand, JointType.RightHand)
        swap_joints(joints, valid_joints, JointType.LeftWaist, JointType.RightWaist)
        swap_joints(joints, valid_joints, JointType.LeftKnee, JointType.RightKnee)
        swap_joints(joints, valid_joints, JointType.LeftFoot, JointType.RightFoot)

        return flipped_img, flipped_mask, joints, valid_joints

    def augment_data(self, orig_img, mask_img, joints, valid_joints, joint_bboxes, min_crop_size):
        """augment data"""

        augmented_img = orig_img.copy()
        augmented_mask = mask_img.copy()
        box_sizes = np.linalg.norm(joint_bboxes[:, 1] - joint_bboxes[:, 0], axis=1)
        min_crop_size = np.min((min(orig_img.shape[:-1]), min_crop_size, int(box_sizes.min() * 5)))
        augmented_img, augmented_mask, joints, valid_joints = self.random_crop_img(augmented_img, augmented_mask, joints, valid_joints, joint_bboxes, min_crop_size)

        # distort color
        augmented_img = self.distort_color(augmented_img)

        # flip image
        if np.random.randint(2):
            augmented_img, augmented_mask, joints, valid_joints = self.flip_img(augmented_img, augmented_mask, joints, valid_joints)

        return augmented_img, augmented_mask, joints, valid_joints

    def compute_heatmaps(self, img, joints, valid_joints, heatmap_sigma):
        """compute heatmaps"""
        heatmaps = np.zeros((0,) + img.shape[:-1])
        sum_heatmap = np.zeros(img.shape[:-1])

        for joint_index in range(len(JointType)):
            heatmap = np.zeros(img.shape[:-1])
            for person_index, person_joints in enumerate(joints):
                if valid_joints[person_index][joint_index]:
                    jointmap = self.gen_gaussian_heatmap(img.shape[:-1], person_joints[joint_index], heatmap_sigma)
                    heatmap[jointmap > heatmap] = jointmap[jointmap > heatmap]
                    sum_heatmap[jointmap > sum_heatmap] = jointmap[jointmap > sum_heatmap]
            heatmaps = np.vstack((heatmaps, heatmap.reshape((1,) + heatmap.shape)))
        bg_heatmap = 1 - sum_heatmap  # background channel
        heatmaps = np.vstack((heatmaps, bg_heatmap[None]))
        return heatmaps.astype('f')

    def compute_pafs(self, img, joints, valid_joints, paf_sigma):
        """compute pafs"""
        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in params['limbs_point']:
            paf = np.zeros((2,) + img.shape[:-1])
            paf_flags = np.zeros(paf.shape) # for constant paf

            for person_index, person_joints in enumerate(joints):
                if valid_joints[person_index][limb[0]] and valid_joints[person_index][limb[1]]:
                    limb_paf = self.gen_constant_paf(img.shape, np.array(person_joints[limb[0]]), np.array(person_joints[limb[1]]), paf_sigma)
                    limb_paf_flags = limb_paf != 0
                    paf_flags += np.broadcast_to(limb_paf_flags[0] | limb_paf_flags[1], limb_paf.shape)
                    paf += limb_paf

            paf[paf_flags > 0] /= paf_flags[paf_flags > 0]
            pafs = np.vstack((pafs, paf))
        return pafs.astype('f')

    def compute_pafs_gaussian(self, img, joints, valid_joints, paf_sigma):
        """compute pafs"""
        pafs = np.zeros((0,) + img.shape[:-1])

        for limb in params['limbs_point']:
            paf = np.zeros((2,) + img.shape[:-1])
            max_paf_length = np.zeros(img.shape[:-1]) # for gaussian paf

            for person_index, person_joints in enumerate(joints):
                if valid_joints[person_index][limb[0]] and valid_joints[person_index][limb[1]]:
                    limb_paf = self.gen_gaussian_paf(img.shape, np.array(person_joints[limb[0]]), np.array(person_joints[limb[1]]), paf_sigma)
                    limb_paf_length = np.sqrt(limb_paf[0] ** 2 + limb_paf[1] ** 2)
                    max_paf_length = np.maximum(max_paf_length, limb_paf_length)
                    paf += limb_paf

            paf_length = np.sqrt(paf[0] ** 2 + paf[1] ** 2)
            paf_length_scale = np.ones(paf_length.shape)
            paf_length_scale[paf_length > 0] = max_paf_length[paf_length > 0] / paf_length[paf_length > 0]
            paf *= np.broadcast_to(paf_length_scale, paf.shape)
            pafs = np.vstack((pafs, paf))

            #cv2.imshow('w', self.overlay_paf(img, paf))
            #cv2.waitKey()
        return pafs

    def parse_coco_annotation(self, img, annotations):
        """coco annotation dataのアノテーションをjoints配列に変換"""
        joints = np.zeros((0, len(JointType), 2), dtype=np.int32)
        valid_joints = np.zeros((0, len(JointType)), dtype=np.bool)
        joint_bboxes = np.zeros((0, 2, 2), np.int32)
        img_mask = np.zeros(img.shape[:-1], dtype=np.uint8)

        for annotation in annotations:
            person_joints = np.zeros((1, len(JointType), 2), dtype=np.int32)
            person_valid_joints = np.zeros((1, len(JointType)), dtype=np.bool)
            person_joint_bbox = np.array([[[np.iinfo(np.int32).max, np.iinfo(np.int32).max], [np.iinfo(np.int32).min, np.iinfo(np.int32).min]]], np.int32)

            # convert joints position
            for i, joint_index in enumerate(params['coco_joint_indices']):
                valid_joint = bool(annotation['keypoints'][i * 3 + 2])
                if valid_joint:
                    person_valid_joints[0][joint_index] = True
                    person_joints[0][joint_index][0] = annotation['keypoints'][i * 3]
                    person_joints[0][joint_index][1] = annotation['keypoints'][i * 3 + 1]
                    person_joint_bbox[0][0][0] = np.minimum(person_joint_bbox[0][0][0],  person_joints[0][joint_index][0]) # left
                    person_joint_bbox[0][0][1] = np.minimum(person_joint_bbox[0][0][1],  person_joints[0][joint_index][1]) # top
                    person_joint_bbox[0][1][0] = np.maximum(person_joint_bbox[0][1][0],  person_joints[0][joint_index][0]) # right
                    person_joint_bbox[0][1][1] = np.maximum(person_joint_bbox[0][1][1],  person_joints[0][joint_index][1]) # bottom

            # compute neck position
            if bool(person_valid_joints[0][JointType.LeftShoulder]) and bool(person_valid_joints[0][JointType.RightShoulder]):
                person_valid_joints[0][JointType.Neck] = True
                person_joints[0][JointType.Neck][0] = int((person_joints[0][JointType.LeftShoulder][0] + person_joints[0][JointType.RightShoulder][0]) / 2)
                person_joints[0][JointType.Neck][1] = int((person_joints[0][JointType.LeftShoulder][1] + person_joints[0][JointType.RightShoulder][1]) / 2)

            # gen mask
            person_mask = self.coco.annToMask(annotation)
            img_mask = img_mask | person_mask

            joints = np.vstack((joints, person_joints))
            valid_joints = np.vstack((valid_joints, person_valid_joints))
            joint_bboxes = np.vstack((joint_bboxes, person_joint_bbox))

        ignore_mask = (~img_mask.astype(np.bool))

        return joints, valid_joints, joint_bboxes, ignore_mask

    def get_sample(self, img, annotations, ignore_mask):
        """get sample data"""
        # params
        crop_size = params['crop_size']
        input_size = params['input_size']
        downscale = params['downscale']
        heatmap_sigma = params['heatmap_sigma']
        paf_sigma = params['paf_sigma']
        downscaled_size = int(input_size / downscale)

        # sample
        joints, valid_joints, joint_bboxes, _ = self.parse_coco_annotation(img, annotations)
        if self.mode != 'eval':
            img, ignore_mask, joints, valid_joints = self.augment_data(img, ignore_mask, joints, valid_joints, joint_bboxes, crop_size)
        resized_img, resized_mask, resized_joints = self.resize_data(img, ignore_mask, joints, resize_shape=(input_size, input_size))
        downscaled_img, downscaled_mask, downscaled_joints = self.resize_data(img, ignore_mask, joints, resize_shape=(downscaled_size, downscaled_size))
        downscaled_heatmaps = self.compute_heatmaps(downscaled_img, downscaled_joints, valid_joints, heatmap_sigma)
        downscaled_pafs = self.compute_pafs(downscaled_img, downscaled_joints, valid_joints, paf_sigma)
        return resized_img, downscaled_pafs, downscaled_heatmaps, downscaled_mask

    def get_img_annotation(self, ind=None, img_id=None):
        """インデックスまたは img_id から coco annotation dataを抽出、条件に満たない場合はNoneを返す """
        annotations = None

        if ind is not None:
            img_id = self.imgIds[ind]
        anno_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)

        # annotation for that image
        if len(anno_ids) > 0:
            annotations_for_img = self.coco.loadAnns(anno_ids)

            person_cnt = 0
            valid_annotations_for_img = []
            for annotation in annotations_for_img:
                # if too few keypoints or too small
                if annotation['num_keypoints'] >= 5 and annotation['area'] > 48 * 48:
                    person_cnt += 1
                    valid_annotations_for_img.append(annotation)

            # if person annotation
            if person_cnt > 0:
                annotations = valid_annotations_for_img

        if self.mode == 'train':
            img_path = os.path.join(params['coco_dir'], 'train2017', self.coco.loadImgs([img_id])[0]['file_name'])
            mask_path = os.path.join(params['coco_dir'], 'ignore_mask_train2017', '{:012d}.png'.format(img_id))
        else:
            img_path = os.path.join(params['coco_dir'], 'val2017', self.coco.loadImgs([img_id])[0]['file_name'])
            mask_path = os.path.join(params['coco_dir'], 'ignore_mask_val2017', '{:012d}.png'.format(img_id))
        img = cv2.imread(img_path)
        ignore_mask = cv2.imread(mask_path, 0)
        if ignore_mask is None:
            ignore_mask = np.zeros(img.shape[:2], 'bool')
        else:
            ignore_mask = ignore_mask == 255
        return img, annotations, ignore_mask, img_id

    def get_example(self, i):
        img, annotations, ignore_mask, img_id = self.get_img_annotation(ind=i)

        if self.mode == 'eval':
            # don't need to make heatmaps/pafs
            return img, annotations, img_id

        # if no annotations are available
        while annotations is None:
            img_id = self.imgIds[np.random.randint(len(self))]
            img, annotations, ignore_mask, img_id = self.get_img_annotation(img_id=img_id)

        resized_img, pafs, heatmaps, ignore_mask = self.get_sample(img, annotations, ignore_mask)
        return resized_img, pafs, heatmaps, ignore_mask

if __name__ == '__main__':
    coco = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_train2017.json'))
    data_loader = CocoDataLoader(coco)

    for i in range(len(data_loader)):
        img, annotations, ignore_mask, img_id = data_loader.get_img_annotation(ind=i)
        if annotations is not None:
            joints, valid_joints, joint_bboxes, _ = data_loader.parse_coco_annotation(img, annotations)
            augmented_img, augmented_mask, joints, valid_joints = data_loader.augment_data(img, ignore_mask, joints, valid_joints, joint_bboxes, min_crop_size=480)

            resized_img, resized_mask, resized_joints = data_loader.resize_data(augmented_img, augmented_mask, joints, resize_shape=(368, 368))
            downscaled_img, downscaled_mask, downscaled_joints = data_loader.resize_data(augmented_img, augmented_mask, joints, resize_shape=(46, 46))

            # compute pafs and heatmaps
            downscaled_pafs = data_loader.compute_pafs(downscaled_img, downscaled_joints, valid_joints, params['paf_sigma'])
            downscaled_heatmaps = data_loader.compute_heatmaps(downscaled_img, downscaled_joints, valid_joints, params['heatmap_sigma'])

            # resize to view
            pafs = cv2.resize(downscaled_pafs.transpose(1, 2, 0), (368, 368)).transpose(2, 0, 1)
            heatmaps = cv2.resize(downscaled_heatmaps.transpose(1, 2, 0), (368, 368)).transpose(2, 0, 1)
            ignore_mask = cv2.resize(downscaled_mask.astype(np.uint8), (368, 368))

            # view
            img = resized_img.copy()
            img = data_loader.overlay_heatmap(img, heatmaps[:-1].max(axis=0))
            img = data_loader.overlay_pafs(img, pafs)
            img = data_loader.overlay_ignore_mask(img, ignore_mask)

            cv2.imshow('w', np.hstack((resized_img, img)))
            k = cv2.waitKey(0)
            if k == ord('q'):
                sys.exit()
