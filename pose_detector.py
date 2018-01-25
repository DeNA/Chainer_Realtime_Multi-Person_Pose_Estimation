import cv2
import math
import time
import argparse
import numpy as np
from scipy.ndimage.filters import gaussian_filter

import chainer
from chainer import cuda, serializers, functions as F

from entity import params, JointType
from models.CocoPoseNet import CocoPoseNet


class PoseDetector(object):
    def __init__(self, arch=None, weights_file=None, model=None, device=-1, precise=False):
        self.arch = arch
        self.precise = precise
        if model is not None:
            self.model = model
        else:
            print('Loading the model...')
            self.model = params['archs'][arch]()

            if weights_file:
                serializers.load_npz(weights_file, self.model)

        self.device = device
        if self.device >= 0:
            cuda.get_device_from_id(device).use()
            self.model.to_gpu()

            # create gaussian filter
            self.gaussian_kernel = self.create_gaussian_kernel(params['gaussian_sigma'], params['ksize'])[None, None]
            self.gaussian_kernel = cuda.to_gpu(self.gaussian_kernel)

    # compute gaussian filter
    def create_gaussian_kernel(self, sigma=1, ksize=5):
        center = int(ksize / 2)
        grid_x = np.tile(np.arange(ksize), (ksize, 1))
        grid_y = grid_x.transpose().copy()
        grid_d2 = (grid_x - center) ** 2 + (grid_y - center) ** 2
        kernel = 1/(sigma**2 * 2 * np.pi) * np.exp(-0.5 * grid_d2 / sigma**2)
        return kernel.astype('f')

    def pad_image(self, img, stride, pad_value):
        h, w, _ = img.shape

        pad = [0] * 2
        pad[0] = (stride - (h % stride)) % stride  # down
        pad[1] = (stride - (w % stride)) % stride  # right

        img_padded = np.zeros((h+pad[0], w+pad[1], 3), 'uint8') + pad_value
        img_padded[:h, :w, :] = img.copy()
        return img_padded, pad

    def compute_optimal_size(self, orig_img, img_size, stride=8):
        """画像の幅と高さがstrideの倍数になるように調節する"""
        orig_img_h, orig_img_w, _ = orig_img.shape
        aspect = orig_img_h / orig_img_w
        if orig_img_h < orig_img_w:
            img_h = img_size
            img_w = np.round(img_size / aspect).astype(int)
            surplus = img_w % stride
            if surplus != 0:
                img_w += stride - surplus
        else:
            img_w = img_size
            img_h = np.round(img_size * aspect).astype(int)
            surplus = img_h % stride
            if surplus != 0:
                img_h += stride - surplus
        return (img_w, img_h)

    def compute_peaks_from_heatmaps(self, heatmaps):
        """all_peaks: shape = [N, 5], column = (jointtype, x, y, score, index)"""

        heatmaps = heatmaps[:-1]

        xp = cuda.get_array_module(heatmaps)

        if xp == np:
            all_peaks = []
            peak_counter = 0
            for i , heatmap in enumerate(heatmaps):
                heatmap = gaussian_filter(heatmap, sigma=params['gaussian_sigma'])
                map_left = xp.zeros(heatmap.shape)
                map_right = xp.zeros(heatmap.shape)
                map_top = xp.zeros(heatmap.shape)
                map_bottom = xp.zeros(heatmap.shape)
                map_left[1:, :] = heatmap[:-1, :]
                map_right[:-1, :] = heatmap[1:, :]
                map_top[:, 1:] = heatmap[:, :-1]
                map_bottom[:, :-1] = heatmap[:, 1:]

                peaks_binary = xp.logical_and.reduce((
                    heatmap > params['heatmap_peak_thresh'],
                    heatmap > map_left,
                    heatmap > map_right,
                    heatmap > map_top,
                    heatmap > map_bottom,
                ))

                peaks = zip(xp.nonzero(peaks_binary)[1], xp.nonzero(peaks_binary)[0])  # [(x, y), (x, y)...]のpeak座標配列
                peaks_with_score = [(i,) + peak_pos + (heatmap[peak_pos[1], peak_pos[0]],) for peak_pos in peaks]
                peaks_id = range(peak_counter, peak_counter + len(peaks_with_score))
                peaks_with_score_and_id = [peaks_with_score[i] + (peaks_id[i], ) for i in range(len(peaks_id))]
                peak_counter += len(peaks_with_score_and_id)
                all_peaks.append(peaks_with_score_and_id)
            all_peaks = xp.array([peak for peaks_each_category in all_peaks for peak in peaks_each_category])
        else:
            heatmaps = F.convolution_2d(heatmaps[:, None], self.gaussian_kernel,
                                        stride=1, pad=int(params['ksize']/2)).data.squeeze()
            left_heatmaps = xp.zeros(heatmaps.shape)
            right_heatmaps = xp.zeros(heatmaps.shape)
            top_heatmaps = xp.zeros(heatmaps.shape)
            bottom_heatmaps = xp.zeros(heatmaps.shape)
            left_heatmaps[:, 1:, :] = heatmaps[:, :-1, :]
            right_heatmaps[:, :-1, :] = heatmaps[:, 1:, :]
            top_heatmaps[:, :, 1:] = heatmaps[:, :, :-1]
            bottom_heatmaps[:, :, :-1] = heatmaps[:, :, 1:]

            peaks_binary = xp.logical_and(heatmaps > params['heatmap_peak_thresh'], heatmaps >= right_heatmaps)
            peaks_binary = xp.logical_and(peaks_binary, heatmaps >= top_heatmaps)
            peaks_binary = xp.logical_and(peaks_binary, heatmaps >= left_heatmaps)
            peaks_binary = xp.logical_and(peaks_binary, heatmaps >= bottom_heatmaps)

            peak_c, peak_y, peak_x = xp.nonzero(peaks_binary)
            peak_score = heatmaps[peak_c, peak_y, peak_x]
            all_peaks = xp.vstack((peak_c, peak_x, peak_y, peak_score)).transpose()
            all_peaks = xp.hstack((all_peaks, xp.arange(len(all_peaks)).reshape(-1, 1)))
            all_peaks = all_peaks.get()
        return all_peaks

    def compute_candidate_connections(self, paf, cand_a, cand_b, img_len, params):
        candidate_connections = []
        for joint_a in cand_a:
            for joint_b in cand_b:  # jointは(x, y)座標
                vector = joint_b[:2] - joint_a[:2]
                norm = np.linalg.norm(vector)
                if norm == 0:
                    continue

                ys = np.linspace(joint_a[1], joint_b[1], num=params['n_integ_points'])
                xs = np.linspace(joint_a[0], joint_b[0], num=params['n_integ_points'])
                integ_points = np.stack([ys, xs]).T.round().astype('i')  # joint_aとjoint_bの2点間を結ぶ線分上の座標点 [[x1, y1], [x2, y2]...]
                paf_in_edge = np.hstack([paf[0][np.hsplit(integ_points, 2)], paf[1][np.hsplit(integ_points, 2)]])
                unit_vector = vector / norm
                inner_products = np.dot(paf_in_edge, unit_vector)

                integ_value = inner_products.sum() / len(inner_products)
                # vectorの長さが基準値以上の時にペナルティを与える
                integ_value_with_dist_prior = integ_value + min(params['limb_length_ratio'] * img_len / norm - params['length_penalty_value'], 0)

                n_valid_points = sum(inner_products > params['inner_product_thresh'])
                if n_valid_points > params['n_integ_points_thresh'] and integ_value_with_dist_prior > 0:
                    candidate_connections.append([int(joint_a[3]), int(joint_b[3]), integ_value_with_dist_prior])
        candidate_connections = sorted(candidate_connections, key=lambda x: x[2], reverse=True)
        return candidate_connections

    def compute_connections(self, pafs, all_peaks, img_len, params):
        all_connections = []
        for i in range(len(params['limbs_point'])):
            paf_index = [i*2, i*2 + 1]
            paf = pafs[paf_index]
            limb_point = params['limbs_point'][i]
            cand_a = all_peaks[all_peaks[:, 0] == limb_point[0]][:, 1:]
            cand_b = all_peaks[all_peaks[:, 0] == limb_point[1]][:, 1:]

            if len(cand_a) > 0 and len(cand_b) > 0:
                candidate_connections = self.compute_candidate_connections(paf, cand_a, cand_b, img_len, params)
                connections = np.zeros((0, 3))
                for index_a, index_b, score in candidate_connections:
                    if index_a not in connections[:, 0] and index_b not in connections[:, 1]:
                        connections = np.vstack([connections, [index_a, index_b, score]])
                        if len(connections) >= min(len(cand_a), len(cand_b)):
                            break
                all_connections.append(connections)
            else:
                all_connections.append(np.zeros((0, 3)))
        return all_connections

    def grouping_key_points(self, all_connections, candidate_peaks, params):
        subsets = -1 * np.ones((0, 20))

        for l, connections in enumerate(all_connections):
            joint_a, joint_b = params['limbs_point'][l]

            for ind_a, ind_b, score in connections[:, :3]:
                ind_a, ind_b = int(ind_a), int(ind_b)

                joint_found_cnt = 0
                joint_found_subset_index = [-1, -1]
                for subset_ind, subset in enumerate(subsets):
                    # そのconnectionのjointをもってるsubsetがいる場合
                    if subset[joint_a] == ind_a or subset[joint_b] == ind_b:
                        joint_found_subset_index[joint_found_cnt] = subset_ind
                        joint_found_cnt += 1

                if joint_found_cnt == 1: # そのconnectionのどちらかのjointをsubsetが持っている場合
                    found_subset = subsets[joint_found_subset_index[0]]
                    # 肩->耳のconnectionの組合せを除いて、始点の一致しか起こり得ない。肩->耳の場合、終点が一致していた場合は、既に顔のbone検出済みなので処理不要。
                    if found_subset[joint_b] != ind_b:
                        found_subset[joint_b] = ind_b
                        found_subset[-1] += 1 # increment joint count
                        found_subset[-2] += candidate_peaks[ind_b, 3] + score  # joint bのscoreとconnectionの積分値を加算

                elif joint_found_cnt == 2: # subset1にjoint1が、subset2にjoint2がある場合(肩->耳のconnectionの組合せした起こり得ない)
                    # print('limb {}: 2 subsets have any joint'.format(l))
                    found_subset_1 = subsets[joint_found_subset_index[0]]
                    found_subset_2 = subsets[joint_found_subset_index[1]]

                    membership = ((found_subset_1 >= 0).astype(int) + (found_subset_2 >= 0).astype(int))[:-2]
                    if not np.any(membership == 2):  # merge two subsets when no duplication
                        found_subset_1[:-2] += found_subset_2[:-2] + 1 # default is -1
                        found_subset_1[-2:] += found_subset_2[-2:]
                        found_subset_1[-2:] += score  # connectionの積分値のみ加算(jointのscoreはmerge時に全て加算済み)
                        subsets = np.delete(subsets, joint_found_subset_index[1], axis=0)
                    else:
                        if found_subset_1[joint_a] == -1:
                            found_subset_1[joint_a] = ind_a
                            found_subset_1[-1] += 1
                            found_subset_1[-2] += candidate_peaks[ind_a, 3] + score
                        elif found_subset_1[joint_b] == -1:
                            found_subset_1[joint_b] = ind_b
                            found_subset_1[-1] += 1
                            found_subset_1[-2] += candidate_peaks[ind_b, 3] + score
                        if found_subset_2[joint_a] == -1:
                            found_subset_2[joint_a] = ind_a
                            found_subset_2[-1] += 1
                            found_subset_2[-2] += candidate_peaks[ind_a, 3] + score
                        elif found_subset_2[joint_b] == -1:
                            found_subset_2[joint_b] = ind_b
                            found_subset_2[-1] += 1
                            found_subset_2[-2] += candidate_peaks[ind_b, 3] + score

                elif joint_found_cnt == 0 and l != 9 and l != 13: # 新規subset作成, 肩耳のconnectionは新規group対象外
                    row = -1 * np.ones(20)
                    row[joint_a] = ind_a
                    row[joint_b] = ind_b
                    row[-1] = 2
                    row[-2] = sum(candidate_peaks[[ind_a, ind_b], 3]) + score
                    subsets = np.vstack([subsets, row])
                elif joint_found_cnt >= 3:
                    pass

        # delete low score subsets
        keep = np.logical_and(subsets[:, -1] >= params['n_subset_limbs_thresh'], subsets[:, -2]/subsets[:, -1] >= params['subset_score_thresh'])
        subsets = subsets[keep]
        return subsets

    def subsets_to_pose_array(self, subsets, all_peaks):
        person_pose_array = []
        for subset in subsets:
            joints = []
            for joint_index in subset[:18].astype('i'):
                if joint_index >= 0:
                    joint = all_peaks[joint_index][1:3].tolist()
                    joint.append(2)
                    joints.append(joint)
                else:
                    joints.append([0, 0, 0])
            person_pose_array.append(np.array(joints))
        person_pose_array = np.array(person_pose_array)
        return person_pose_array

    def compute_limbs_length(self, joints):
        limbs = []
        limbs_len = np.zeros(len(params["limbs_point"]))
        for i, joint_indices in enumerate(params["limbs_point"]):
            if joints[joint_indices[0]] is not None and joints[joint_indices[1]] is not None:
                limbs.append([joints[joint_indices[0]], joints[joint_indices[1]]])
                limbs_len[i] = np.linalg.norm(joints[joint_indices[1]][:-1] - joints[joint_indices[0]][:-1])
            else:
                limbs.append(None)

        return limbs_len, limbs

    def compute_unit_length(self, limbs_len):
        unit_length = 0
        base_limbs_len = limbs_len[[14, 3, 0, 13, 9]] # (鼻首、首左腰、首右腰、肩左耳、肩右耳)の長さの比率(このどれかが存在すればこれを優先的に単位長さの計算する)
        non_zero_limbs_len = base_limbs_len > 0
        if len(np.nonzero(non_zero_limbs_len)[0]) > 0:
            limbs_len_ratio = np.array([0.85, 2.2, 2.2, 0.85, 0.85])
            unit_length = np.sum(base_limbs_len[non_zero_limbs_len] / limbs_len_ratio[non_zero_limbs_len]) / len(np.nonzero(non_zero_limbs_len)[0])
        else:
            limbs_len_ratio = np.array([2.2, 1.7, 1.7, 2.2, 1.7, 1.7, 0.6, 0.93, 0.65, 0.85, 0.6, 0.93, 0.65, 0.85, 1, 0.2, 0.2, 0.25, 0.25])
            non_zero_limbs_len = limbs_len > 0
            unit_length = np.sum(limbs_len[non_zero_limbs_len] / limbs_len_ratio[non_zero_limbs_len]) / len(np.nonzero(non_zero_limbs_len)[0])

        return unit_length

    def get_unit_length(self, person_pose):
        limbs_length, limbs = self.compute_limbs_length(person_pose)
        unit_length = self.compute_unit_length(limbs_length)

        return unit_length

    def crop_around_keypoint(self, img, keypoint, crop_size):
        x, y = keypoint
        left = int(x - crop_size)
        top = int(y - crop_size)
        right = int(x + crop_size)
        bottom = int(y + crop_size)
        bbox = (left, top, right, bottom)

        cropped_img = self.crop_image(img, bbox)

        return cropped_img, bbox

    def crop_person(self, img, person_pose, unit_length):
        top_joint_priority = [4, 5, 6, 12, 16, 7, 13, 17, 8, 10, 14, 9, 11, 15, 2, 3, 0, 1, sys.maxsize]
        bottom_joint_priority = [9, 6, 7, 14, 16, 8, 15, 17, 4, 2, 0, 5, 3, 1, 10, 11, 12, 13, sys.maxsize]

        top_joint_index = len(top_joint_priority) - 1
        bottom_joint_index = len(bottom_joint_priority) - 1
        left_joint_index = 0
        right_joint_index = 0
        top_pos = sys.maxsize
        bottom_pos = 0
        left_pos = sys.maxsize
        right_pos = 0

        for i, joint in enumerate(person_pose):
            if joint[2] > 0:
                if top_joint_priority[i] < top_joint_priority[top_joint_index]:
                    top_joint_index = i
                elif bottom_joint_priority[i] < bottom_joint_priority[bottom_joint_index]:
                    bottom_joint_index = i
                if joint[1] < top_pos:
                    top_pos = joint[1]
                elif joint[1] > bottom_pos:
                    bottom_pos = joint[1]

                if joint[0] < left_pos:
                    left_pos = joint[0]
                    left_joint_index = i
                elif joint[0] > right_pos:
                    right_pos = joint[0]
                    right_joint_index = i

        top_padding_radio = [0.9, 1.9, 1.9, 2.9, 3.7, 1.9, 2.9, 3.7, 4.0, 5.5, 7.0, 4.0, 5.5, 7.0, 0.7, 0.8, 0.7, 0.8]
        bottom_padding_radio = [6.9, 5.9, 5.9, 4.9, 4.1, 5.9, 4.9, 4.1, 3.8, 2.3, 0.8, 3.8, 2.3, 0.8, 7.1, 7.0, 7.1, 7.0]

        left = (left_pos - 0.3 * unit_length).astype(int)
        right = (right_pos + 0.3 * unit_length).astype(int)
        top = (top_pos - top_padding_radio[top_joint_index] * unit_length).astype(int)
        bottom = (bottom_pos + bottom_padding_radio[bottom_joint_index] * unit_length).astype(int)
        bbox = (left, top, right, bottom)

        cropped_img = self.crop_image(img, bbox)
        return cropped_img, bbox

    def crop_face(self, img, person_pose, unit_length):
        face_size = unit_length
        face_img = None
        bbox = None

        # if have nose
        if person_pose[JointType.Nose][2] > 0:
            nose_pos = person_pose[JointType.Nose][:2]
            face_top = int(nose_pos[1] - face_size * 1.2)
            face_bottom = int(nose_pos[1] + face_size * 0.8)
            face_left = int(nose_pos[0] - face_size)
            face_right = int(nose_pos[0] + face_size)
            bbox = (face_left, face_top, face_right, face_bottom)
            face_img = self.crop_image(img, bbox)

        return face_img, bbox

    def crop_hands(self, img, person_pose, unit_length):
        hands = {
            "left": None,
            "right": None
        }

        if person_pose[JointType.LeftHand][2] > 0:
            crop_center = person_pose[JointType.LeftHand][:-1]
            if person_pose[JointType.LeftElbow][2] > 0:
                direction_vec = person_pose[JointType.LeftHand][:-1] - person_pose[JointType.LeftElbow][:-1]
                crop_center += (0.3 * direction_vec).astype(crop_center.dtype)
            hand_img, bbox = self.crop_around_keypoint(img, crop_center, unit_length * 0.95)
            hands["left"] = {
                "img": hand_img,
                "bbox": bbox
            }

        if person_pose[JointType.RightHand][2] > 0:
            crop_center = person_pose[JointType.RightHand][:-1]
            if person_pose[JointType.RightElbow][2] > 0:
                direction_vec = person_pose[JointType.RightHand][:-1] - person_pose[JointType.RightElbow][:-1]
                crop_center += (0.3 * direction_vec).astype(crop_center.dtype)
            hand_img, bbox = self.crop_around_keypoint(img, crop_center, unit_length * 0.95)
            hands["right"] = {
                "img": hand_img,
                "bbox": bbox
            }

        return hands

    def crop_image(self, img, bbox):
        left, top, right, bottom = bbox
        img_h, img_w, img_ch = img.shape
        box_h = bottom - top
        box_w = right - left

        crop_left = max(0, left)
        crop_top = max(0, top)
        crop_right = min(img_w, right)
        crop_bottom = min(img_h, bottom)
        crop_h = crop_bottom - crop_top
        crop_w = crop_right - crop_left
        cropped_img = img[crop_top:crop_bottom, crop_left:crop_right]

        bias_x = bias_y = 0
        if left < crop_left:
            bias_x = crop_left - left
        if top < crop_top:
            bias_y = crop_top - top

        # pad
        padded_img = np.zeros((box_h, box_w, img_ch), dtype=np.uint8)
        padded_img[bias_y:bias_y+crop_h, bias_x:bias_x+crop_w] = cropped_img
        return padded_img

    def preprocess(self, img):
        x_data = img.astype('f')
        x_data /= 255
        x_data -= 0.5
        x_data = x_data.transpose(2, 0, 1)[None]
        return x_data

    def detect_precise(self, orig_img):
        orig_img_h, orig_img_w, _ = orig_img.shape

        pafs_sum = 0
        heatmaps_sum = 0

        interpolation = cv2.INTER_CUBIC

        for scale in params['inference_scales']:
            multiplier = scale * params['inference_img_size'] / min(orig_img.shape[:2])
            img = cv2.resize(orig_img, (math.ceil(orig_img_w*multiplier), math.ceil(orig_img_h*multiplier)), interpolation=interpolation)
            bbox = (params['inference_img_size'], max(params['inference_img_size'], img.shape[1]))
            padded_img, pad = self.pad_image(img, params['downscale'], (104, 117, 123))

            x_data = self.preprocess(padded_img)
            if self.device >= 0:
                x_data = cuda.to_gpu(x_data)

            h1s, h2s = self.model(x_data)

            tmp_paf = h1s[-1][0].data.transpose(1, 2, 0)
            tmp_heatmap = h2s[-1][0].data.transpose(1, 2, 0)

            if self.device >= 0:
                tmp_paf = cuda.to_cpu(tmp_paf)
                tmp_heatmap = cuda.to_cpu(tmp_heatmap)

            p_h, p_w = padded_img.shape[:2]
            tmp_paf = cv2.resize(tmp_paf, (p_w, p_h), interpolation=interpolation)
            tmp_paf = tmp_paf[:p_h-pad[0], :p_w-pad[1], :]
            pafs_sum += cv2.resize(tmp_paf, (orig_img_w, orig_img_h), interpolation=interpolation)

            tmp_heatmap = cv2.resize(tmp_heatmap, (0, 0), fx=params['downscale'], fy=params['downscale'], interpolation=interpolation)
            tmp_heatmap = tmp_heatmap[:padded_img.shape[0]-pad[0], :padded_img.shape[1]-pad[1], :]
            heatmaps_sum += cv2.resize(tmp_heatmap, (orig_img_w, orig_img_h), interpolation=interpolation)

        self.pafs = (pafs_sum / len(params['inference_scales'])).transpose(2, 0, 1)
        self.heatmaps = (heatmaps_sum / len(params['inference_scales'])).transpose(2, 0, 1)

        if self.device >= 0:
            self.pafs = cuda.to_cpu(self.pafs)

        self.all_peaks = self.compute_peaks_from_heatmaps(self.heatmaps)
        if len(self.all_peaks) == 0:
            return np.empty((0, len(JointType), 3)), np.empty(0)
        all_connections = self.compute_connections(self.pafs, self.all_peaks, orig_img_w, params)
        subsets = self.grouping_key_points(all_connections, self.all_peaks, params)
        poses = self.subsets_to_pose_array(subsets, self.all_peaks)
        scores = subsets[:, -2]
        return poses, scores

    def __call__(self, orig_img):
        orig_img = orig_img.copy()
        if self.precise:
            return self.detect_precise(orig_img)
        orig_img_h, orig_img_w, _ = orig_img.shape

        input_w, input_h = self.compute_optimal_size(orig_img, params['inference_img_size'])
        map_w, map_h = self.compute_optimal_size(orig_img, params['heatmap_size'])

        resized_image = cv2.resize(orig_img, (input_w, input_h))
        x_data = self.preprocess(resized_image)

        if self.device >= 0:
            x_data = cuda.to_gpu(x_data)

        h1s, h2s = self.model(x_data)

        pafs = F.resize_images(h1s[-1], (map_h, map_w)).data[0]
        heatmaps = F.resize_images(h2s[-1], (map_h, map_w)).data[0]

        if self.device >= 0:
            pafs = pafs.get()
            cuda.get_device_from_id(self.device).synchronize()

        all_peaks = self.compute_peaks_from_heatmaps(heatmaps)
        if len(all_peaks) == 0:
            return np.empty((0, len(JointType), 3)), np.empty(0)
        all_connections = self.compute_connections(pafs, all_peaks, map_w, params)
        subsets = self.grouping_key_points(all_connections, all_peaks, params)
        all_peaks[:, 1] *= orig_img_w / map_w
        all_peaks[:, 2] *= orig_img_h / map_h
        poses = self.subsets_to_pose_array(subsets, all_peaks)
        scores = subsets[:, -2]
        return poses, scores


def draw_person_pose(orig_img, poses):
    if len(poses) == 0:
        return orig_img

    limb_colors = [
        [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],
        [0, 85, 255], [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0.],
        [255, 0, 85], [170, 255, 0], [85, 255, 0], [170, 0, 255.], [0, 0, 255],
        [0, 0, 255], [255, 0, 255], [170, 0, 255], [255, 0, 170],
    ]

    joint_colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255],
        [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    canvas = orig_img.copy()

    # limbs
    for pose in poses.round().astype('i'):
        for i, (limb, color) in enumerate(zip(params['limbs_point'], limb_colors)):
            if i != 9 and i != 13:  # don't show ear-shoulder connection
                limb_ind = np.array(limb)
                if np.all(pose[limb_ind][:, 2] != 0):
                    joint1, joint2 = pose[limb_ind][:, :2]
                    cv2.line(canvas, tuple(joint1), tuple(joint2), color, 2)

    # joints
    for pose in poses.round().astype('i'):
        for i, ((x, y, v), color) in enumerate(zip(pose, joint_colors)):
            if v != 0:
                cv2.circle(canvas, (x, y), 3, color, -1)
    return canvas

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pose detector')
    parser.add_argument('arch', choices=params['archs'].keys(), default='posenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--img', '-i', default=None, help='image file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--precise', action='store_true', help='visualize results')
    args = parser.parse_args()

    chainer.config.enable_backprop = False
    chainer.config.train = False

    # load model
    pose_detector = PoseDetector(args.arch, args.weights, device=args.gpu, precise=args.precise)

    # read image
    img = cv2.imread(args.img)

    # inference
    poses, _ = pose_detector(img)

    # draw and save image
    img = draw_person_pose(img, poses)
    print('Saving result into result.png...')
    cv2.imwrite('result.png', img)
