import cv2
import argparse
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import chainer
from chainer import cuda, serializers, functions as F
from entity import params
from models.HandNet import HandNet

chainer.using_config('enable_backprop', False)

class HandDetector(object):
    def __init__(self, arch=None, weights_file=None, model=None, device=-1):
        print('Loading HandNet...')
        self.model = params['archs'][arch]()
        serializers.load_npz(weights_file, self.model)

        self.device = device
        if self.device >= 0:
            cuda.get_device_from_id(device).use()
            self.model.to_gpu()

            # create gaussian filter
            ksize = params['ksize']
            kernel = cuda.to_gpu(self.create_gaussian_kernel(sigma=params['gaussian_sigma'], ksize=ksize))
            self.gaussian_kernel = kernel

    def __call__(self, hand_img, fast_mode=False):
        hand_img_h, hand_img_w, _ = hand_img.shape

        resized_image = cv2.resize(hand_img, (params["hand_inference_img_size"], params["hand_inference_img_size"]))
        x_data = np.array(resized_image[np.newaxis], dtype=np.float32).transpose(0, 3, 1, 2) / 256 - 0.5

        if self.device >= 0:
            x_data = cuda.to_gpu(x_data)

        hs = self.model(x_data)
        heatmaps = F.resize_images(hs[-1], (hand_img_h, hand_img_w)).data[0]
        keypoints = self.compute_peaks_from_heatmaps(heatmaps)

        return keypoints

    # compute gaussian filter
    def create_gaussian_kernel(self, sigma=1, ksize=5):
        center = int(ksize / 2)
        kernel = np.zeros((1, 1, ksize, ksize), dtype=np.float32)
        for y in range(ksize):
            distance_y = abs(y-center)
            for x in range(ksize):
                distance_x = abs(x-center)
                kernel[0][0][y][x] = 1/(sigma**2 * 2 * np.pi) * np.exp(-(distance_x**2 + distance_y**2)/(2 * sigma**2))
        return kernel

    def compute_peaks_from_heatmaps(self, heatmaps):
        keypoints = []
        xp = cuda.get_array_module(heatmaps)

        if xp == np:
            for i in range(heatmaps.shape[0] - 1):
                heatmap = gaussian_filter(heatmaps[i], sigma=params['gaussian_sigma'])
                max_value = heatmap.max()
                if max_value > params['hand_heatmap_peak_thresh']:
                    coords = np.array(np.where(heatmap==max_value)).flatten().tolist()
                    keypoints.append([coords[1], coords[0], max_value]) # x, y, conf
                else:
                    keypoints.append(None)
        else:
            heatmaps = F.convolution_2d(heatmaps[:, None], self.gaussian_kernel, stride=1, pad=int(params['ksize']/2)).data.squeeze().get()
            for heatmap in heatmaps[:-1]:
                max_value = heatmap.max()
                if max_value > params['hand_heatmap_peak_thresh']:
                    coords = np.array(np.where(heatmap==max_value)).flatten().tolist()
                    keypoints.append([coords[1], coords[0], max_value]) # x, y, conf
                else:
                    keypoints.append(None)

        return keypoints

def draw_hand_keypoints(orig_img, hand_keypoints, left_top):
    img = orig_img.copy()
    left, top = left_top

    finger_colors = [
        (0, 0, 255),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
    ]

    for i, finger_indices in enumerate(params["fingers_indices"]):
        for finger_line_index in finger_indices:
            keypoint_from = hand_keypoints[finger_line_index[0]]
            keypoint_to = hand_keypoints[finger_line_index[1]]

            if keypoint_from:
                keypoint_from_x, keypoint_from_y, _ = keypoint_from
                cv2.circle(img, (keypoint_from_x + left, keypoint_from_y + top), 3, finger_colors[i], -1)

            if keypoint_to:
                keypoint_to_x, keypoint_to_y, _ = keypoint_to
                cv2.circle(img, (keypoint_to_x + left, keypoint_to_y + top), 3, finger_colors[i], -1)

            if keypoint_from and keypoint_to:
                cv2.line(img, (keypoint_from_x + left, keypoint_from_y + top), (keypoint_to_x + left, keypoint_to_y + top), finger_colors[i], 1)

    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face detector')
    parser.add_argument('arch', choices=params['archs'].keys(), default='facenet', help='Model architecture')
    parser.add_argument('weights', help='weights file path')
    parser.add_argument('--img', help='image file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    # load model
    hand_detector = HandDetector(args.arch, args.weights, device=args.gpu)

    # read image
    img = cv2.imread(args.img)

    # inference
    hand_keypoints = hand_detector(img)

    # draw and save image
    img = draw_hand_keypoints(img, hand_keypoints, (0, 0))
    print('Saving result into result.png...')
    cv2.imwrite('result.png', img)
