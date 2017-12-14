import os
import cv2
import copy
import json
import glob
import random
import argparse
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import chainer
from chainer import cuda, training, reporter, function
from chainer.training import StandardUpdater, extensions
from chainer import serializers, optimizers, functions as F

from entity import JointType, params
from coco_data_loader import CocoDataLoader
from pose_detector import PoseDetector, draw_person_pose

from models import CocoPoseNet


def compute_loss(pafs_ys, heatmaps_ys, pafs_t, heatmaps_t, ignore_mask):
    heatmaps_loss_log = []
    pafs_loss_log = []
    total_loss = 0

    paf_masks = ignore_mask[:, None].repeat(pafs_t.shape[1], axis=1)
    heatmap_masks = ignore_mask[:, None].repeat(heatmaps_t.shape[1], axis=1)

    for pafs_y, heatmaps_y in zip(pafs_ys, heatmaps_ys): # compute loss on each stage
        # consider mask on each stage
        stage_pafs_t = pafs_t.copy()
        stage_heatmaps_t = heatmaps_t.copy()

        stage_pafs_t[paf_masks == True] = pafs_y.data[paf_masks == True]
        stage_heatmaps_t[heatmap_masks == True] = heatmaps_y.data[heatmap_masks == True]

        pafs_loss = F.mean_squared_error(stage_pafs_t, pafs_y)
        heatmaps_loss = F.mean_squared_error(stage_heatmaps_t, heatmaps_y)
        total_loss += pafs_loss + heatmaps_loss

        pafs_loss_log.append(float(cuda.to_cpu(pafs_loss.data)))
        heatmaps_loss_log.append(float(cuda.to_cpu(heatmaps_loss.data)))
    return total_loss, pafs_loss_log, heatmaps_loss_log


def preprocess(imgs):
    xp = cuda.get_array_module(imgs)
    x_data = imgs.astype('f')
    x_data /= 255
    x_data -= 0.5
    x_data = x_data.transpose(0, 3, 1, 2)
    return x_data


class Updater(StandardUpdater):

    def __init__(self, iterator, model, optimizer, device=None):
        super(Updater, self).__init__(
            iterator, optimizer, device=device)

    def update_core(self):
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        batch = train_iter.next()
        imgs, pafs, heatmaps, ignore_mask = self.converter(batch, self.device)

        x_data = preprocess(imgs)

        inferenced_pafs, inferenced_heatmaps = optimizer.target(x_data)

        loss, pafs_loss_log, heatmaps_loss_log = compute_loss(
            inferenced_pafs, inferenced_heatmaps, pafs, heatmaps, ignore_mask)

        reporter.report({
            'main/loss': loss,
            'paf/loss': sum(pafs_loss_log),
            'map/loss': sum(heatmaps_loss_log),
        })

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()


class Evaluator(extensions.Evaluator):

    def __init__(self, iterator, model, device=None):
        super(Evaluator, self).__init__(iterator, model, device=device)
        self.iterator = iterator

    def evaluate(self):
        val_iter = self.get_iterator('main')
        model = self.get_target('main')

        it = copy.copy(val_iter)

        summary = reporter.DictSummary()
        res = []
        for i, batch in enumerate(it):
            observation = {}
            with reporter.report_scope(observation):
                imgs, pafs, heatmaps, ignore_mask = self.converter(batch, self.device)
                with function.no_backprop_mode():

                    x_data = preprocess(imgs)

                    inferenced_pafs, inferenced_heatmaps = model(x_data)

                    loss, pafs_loss_log, heatmaps_loss_log = compute_loss(
                        inferenced_pafs, inferenced_heatmaps, pafs, heatmaps, ignore_mask)
                    observation['val/loss'] = cuda.to_cpu(loss.data)
            summary.add(observation)
        return summary.compute_mean()


def parse_args():
    parser = argparse.ArgumentParser(description='Train pose estimation')
    parser.add_argument('--arch', '-a', choices=params['archs'].keys(), default='posenet',
                        help='Model architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=10,
                        help='Learning minibatch size')
    parser.add_argument('--valbatchsize', '-b', type=int, default=10,
                        help='Validation minibatch size')
    parser.add_argument('--val_samples', type=int, default=100,
                        help='Number of validation samples')
    parser.add_argument('--iteration', '-i', type=int, default=600000,
                        help='Number of iterations to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result/test',
                        help='Output directory')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    model = params['archs'][args.arch]()
    if args.arch == 'posenet':
        CocoPoseNet.copy_vgg_params(model)

    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Load the datasets
    coco_train = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_train2017.json'))
    coco_val = COCO(os.path.join(params['coco_dir'], 'annotations/person_keypoints_val2017.json'))
    train_loader = CocoDataLoader(coco_train, mode='train')
    val_loader = CocoDataLoader(coco_val, mode='val', n_samples=args.val_samples)

    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    multiprocessing.set_start_method('spawn')  # to avoid MultiprocessIterator's bug
    train_iter = chainer.iterators.MultiprocessIterator(
        train_loader, args.batchsize, n_processes=args.loaderjob)
    val_iter = chainer.iterators.MultiprocessIterator(
        val_loader, args.valbatchsize, n_processes=args.loaderjob, repeat=False, shuffle=False)

    # Set up an optimizer
    # optimizer = optimizers.MomentumSGD(lr=4e-5, momentum=0.9)
    optimizer = optimizers.Adam(alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-08)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # Set up a trainer
    updater = Updater(train_iter, model, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.iteration, 'iteration'), args.out)

    val_interval = (2 if args.test else 1000), 'iteration'
    log_interval = (1 if args.test else 10), 'iteration'

    trainer.extend(Evaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'val/loss',
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()
