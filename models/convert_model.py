from CocoPoseNet import CocoPoseNet
from FaceNet import FaceNet
from HandNet import HandNet
from chainer import serializers
from chainer.links import caffe
import argparse

layer_names = {

    "posenet": [
        "conv1_1",
        "conv1_2",
        "conv2_1",
        "conv2_2",
        "conv3_1",
        "conv3_2",
        "conv3_3",
        "conv3_4",
        "conv4_1",
        "conv4_2",
        "conv4_3_CPM",
        "conv4_4_CPM",

        # stage1
        "conv5_1_CPM_L1",
        "conv5_2_CPM_L1",
        "conv5_3_CPM_L1",
        "conv5_4_CPM_L1",
        "conv5_1_CPM_L2",
        "conv5_2_CPM_L2",
        "conv5_3_CPM_L2",
        "conv5_4_CPM_L2",
        "conv5_5_CPM_L2",

        # stage2
        "Mconv1_stage2_L1",
        "Mconv2_stage2_L1",
        "Mconv3_stage2_L1",
        "Mconv4_stage2_L1",
        "Mconv5_stage2_L1",
        "Mconv6_stage2_L1",
        "Mconv7_stage2_L1",
        "Mconv1_stage2_L2",
        "Mconv2_stage2_L2",
        "Mconv3_stage2_L2",
        "Mconv4_stage2_L2",
        "Mconv5_stage2_L2",
        "Mconv6_stage2_L2",
        "Mconv7_stage2_L2",

        # stage3
        "Mconv1_stage3_L1",
        "Mconv2_stage3_L1",
        "Mconv3_stage3_L1",
        "Mconv4_stage3_L1",
        "Mconv5_stage3_L1",
        "Mconv6_stage3_L1",
        "Mconv7_stage3_L1",
        "Mconv1_stage3_L2",
        "Mconv2_stage3_L2",
        "Mconv3_stage3_L2",
        "Mconv4_stage3_L2",
        "Mconv5_stage3_L2",
        "Mconv6_stage3_L2",
        "Mconv7_stage3_L2",

        # stage4
        "Mconv1_stage4_L1",
        "Mconv2_stage4_L1",
        "Mconv3_stage4_L1",
        "Mconv4_stage4_L1",
        "Mconv5_stage4_L1",
        "Mconv6_stage4_L1",
        "Mconv7_stage4_L1",
        "Mconv1_stage4_L2",
        "Mconv2_stage4_L2",
        "Mconv3_stage4_L2",
        "Mconv4_stage4_L2",
        "Mconv5_stage4_L2",
        "Mconv6_stage4_L2",
        "Mconv7_stage4_L2",

        # stage5
        "Mconv1_stage5_L1",
        "Mconv2_stage5_L1",
        "Mconv3_stage5_L1",
        "Mconv4_stage5_L1",
        "Mconv5_stage5_L1",
        "Mconv6_stage5_L1",
        "Mconv7_stage5_L1",
        "Mconv1_stage5_L2",
        "Mconv2_stage5_L2",
        "Mconv3_stage5_L2",
        "Mconv4_stage5_L2",
        "Mconv5_stage5_L2",
        "Mconv6_stage5_L2",
        "Mconv7_stage5_L2",

        # stage6
        "Mconv1_stage6_L1",
        "Mconv2_stage6_L1",
        "Mconv3_stage6_L1",
        "Mconv4_stage6_L1",
        "Mconv5_stage6_L1",
        "Mconv6_stage6_L1",
        "Mconv7_stage6_L1",
        "Mconv1_stage6_L2",
        "Mconv2_stage6_L2",
        "Mconv3_stage6_L2",
        "Mconv4_stage6_L2",
        "Mconv5_stage6_L2",
        "Mconv6_stage6_L2",
        "Mconv7_stage6_L2",
    ],

    "facenet": [
        "conv1_1",
        "conv1_2",
        "conv2_1",
        "conv2_2",
        "conv3_1",
        "conv3_2",
        "conv3_3",
        "conv3_4",
        "conv4_1",
        "conv4_2",
        "conv4_3",
        "conv4_4",
        "conv5_1",
        "conv5_2",
        "conv5_3_CPM",

        # stage1
        "conv6_1_CPM",
        "conv6_2_CPM",

        # stage2
        "Mconv1_stage2",
        "Mconv2_stage2",
        "Mconv3_stage2",
        "Mconv4_stage2",
        "Mconv5_stage2",
        "Mconv6_stage2",
        "Mconv7_stage2",

        # stage3
        "Mconv1_stage3",
        "Mconv2_stage3",
        "Mconv3_stage3",
        "Mconv4_stage3",
        "Mconv5_stage3",
        "Mconv6_stage3",
        "Mconv7_stage3",

        # stage4
        "Mconv1_stage4",
        "Mconv2_stage4",
        "Mconv3_stage4",
        "Mconv4_stage4",
        "Mconv5_stage4",
        "Mconv6_stage4",
        "Mconv7_stage4",

        # stage5
        "Mconv1_stage5",
        "Mconv2_stage5",
        "Mconv3_stage5",
        "Mconv4_stage5",
        "Mconv5_stage5",
        "Mconv6_stage5",
        "Mconv7_stage5",

        # stage6
        "Mconv1_stage6",
        "Mconv2_stage6",
        "Mconv3_stage6",
        "Mconv4_stage6",
        "Mconv5_stage6",
        "Mconv6_stage6",
        "Mconv7_stage6",
    ],

    "handnet": [
        "conv1_1",
        "conv1_2",
        "conv2_1",
        "conv2_2",
        "conv3_1",
        "conv3_2",
        "conv3_3",
        "conv3_4",
        "conv4_1",
        "conv4_2",
        "conv4_3",
        "conv4_4",
        "conv5_1",
        "conv5_2",
        "conv5_3_CPM",

        # stage1
        "conv6_1_CPM",
        "conv6_2_CPM",

        # stage2
        "Mconv1_stage2",
        "Mconv2_stage2",
        "Mconv3_stage2",
        "Mconv4_stage2",
        "Mconv5_stage2",
        "Mconv6_stage2",
        "Mconv7_stage2",

        # stage3
        "Mconv1_stage3",
        "Mconv2_stage3",
        "Mconv3_stage3",
        "Mconv4_stage3",
        "Mconv5_stage3",
        "Mconv6_stage3",
        "Mconv7_stage3",

        # stage4
        "Mconv1_stage4",
        "Mconv2_stage4",
        "Mconv3_stage4",
        "Mconv4_stage4",
        "Mconv5_stage4",
        "Mconv6_stage4",
        "Mconv7_stage4",

        # stage5
        "Mconv1_stage5",
        "Mconv2_stage5",
        "Mconv3_stage5",
        "Mconv4_stage5",
        "Mconv5_stage5",
        "Mconv6_stage5",
        "Mconv7_stage5",

        # stage6
        "Mconv1_stage6",
        "Mconv2_stage6",
        "Mconv3_stage6",
        "Mconv4_stage6",
        "Mconv5_stage6",
        "Mconv6_stage6",
        "Mconv7_stage6",
    ],
}

models = {
    'posenet': CocoPoseNet,
    'facenet': FaceNet,
    'handnet': HandNet,
}

def copy_conv_layer_weights(chainer_model, caffe_model, layer_name):
    if eval("chainer_model.%s.b.shape == caffe_model['%s'].b.shape" % (layer_name, layer_name)) and eval("chainer_model.%s.W.shape == caffe_model['%s'].W.shape" % (layer_name, layer_name)):
        exec("chainer_model.%s.W.data = caffe_model['%s'].W.data" % (layer_name, layer_name))
        exec("chainer_model.%s.b.data = caffe_model['%s'].b.data" % (layer_name, layer_name))
        print("Succeed to copy layer %s" % (layer_name))
    else:
        print("Failed to copy layer %s!" % (layer_name))

parser = argparse.ArgumentParser(description="Convert caffemodel into chainermodel")
parser.add_argument("arch", help="model architecture: ['posenet', 'facenet', 'handnet']")
parser.add_argument("caffe_file", help="caffe weights file path")
parser.add_argument("chainer_file", help="file path to save chainer weights file")
args = parser.parse_args()

print("Loading PoseNet...")
chainer_model = models[args.arch]()

print("Loading caffemodel file...")
caffe_model = caffe.CaffeFunction(args.caffe_file)

for layer_name in layer_names[args.arch]:
    copy_conv_layer_weights(chainer_model, caffe_model, layer_name)

print("Saving weights file into '%s'..." % (args.chainer_file))
serializers.save_npz(args.chainer_file, chainer_model)
print("Done.")
