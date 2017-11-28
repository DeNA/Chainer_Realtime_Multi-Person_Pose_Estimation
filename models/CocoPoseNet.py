import chainer
import chainer.functions as F
import chainer.links as L
from chainer.links import caffe


def copy_vgg_params(model):
    print('Copying params of pretrained model...')
    layer_names = [
        "conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1",
        "conv3_2", "conv3_3", "conv3_4", "conv4_1", "conv4_2",
    ]
    pre_model = caffe.CaffeFunction('models/VGG_ILSVRC_19_layers.caffemodel')
    for layer_name in layer_names:
        exec("model.%s.W.data = pre_model['%s'].W.data" % (layer_name, layer_name))
        exec("model.%s.b.data = pre_model['%s'].b.data" % (layer_name, layer_name))
    print('Done.')


class CocoPoseNet(chainer.Chain):
    insize = 368

    def __init__(self):
        super(CocoPoseNet, self).__init__(
            # cnn to make feature map
            conv1_1=L.Convolution2D(in_channels=3, out_channels=64, ksize=3, stride=1, pad=1),
            conv1_2=L.Convolution2D(in_channels=64, out_channels=64, ksize=3, stride=1, pad=1),
            conv2_1=L.Convolution2D(in_channels=64, out_channels=128, ksize=3, stride=1, pad=1),
            conv2_2=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv3_1=L.Convolution2D(in_channels=128, out_channels=256, ksize=3, stride=1, pad=1),
            conv3_2=L.Convolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=1),
            conv3_3=L.Convolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=1),
            conv3_4=L.Convolution2D(in_channels=256, out_channels=256, ksize=3, stride=1, pad=1),
            conv4_1=L.Convolution2D(in_channels=256, out_channels=512, ksize=3, stride=1, pad=1),
            conv4_2=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1, pad=1),
            conv4_3_CPM=L.Convolution2D(in_channels=512, out_channels=256, ksize=3, stride=1, pad=1),
            conv4_4_CPM=L.Convolution2D(in_channels=256, out_channels=128, ksize=3, stride=1, pad=1),

            # stage1
            conv5_1_CPM_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_2_CPM_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_3_CPM_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_4_CPM_L1=L.Convolution2D(in_channels=128, out_channels=512, ksize=1, stride=1, pad=0),
            conv5_5_CPM_L1=L.Convolution2D(in_channels=512, out_channels=38, ksize=1, stride=1, pad=0),
            conv5_1_CPM_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_2_CPM_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_3_CPM_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=3, stride=1, pad=1),
            conv5_4_CPM_L2=L.Convolution2D(in_channels=128, out_channels=512, ksize=1, stride=1, pad=0),
            conv5_5_CPM_L2=L.Convolution2D(in_channels=512, out_channels=19, ksize=1, stride=1, pad=0),

            # stage2
            Mconv1_stage2_L1=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage2_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage2_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage2_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage2_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage2_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage2_L1=L.Convolution2D(in_channels=128, out_channels=38, ksize=1, stride=1, pad=0),
            Mconv1_stage2_L2=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage2_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage2_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage2_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage2_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage2_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage2_L2=L.Convolution2D(in_channels=128, out_channels=19, ksize=1, stride=1, pad=0),

            # stage3
            Mconv1_stage3_L1=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage3_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage3_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage3_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage3_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage3_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage3_L1=L.Convolution2D(in_channels=128, out_channels=38, ksize=1, stride=1, pad=0),
            Mconv1_stage3_L2=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage3_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage3_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage3_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage3_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage3_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage3_L2=L.Convolution2D(in_channels=128, out_channels=19, ksize=1, stride=1, pad=0),

            # stage4
            Mconv1_stage4_L1=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage4_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage4_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage4_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage4_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage4_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage4_L1=L.Convolution2D(in_channels=128, out_channels=38, ksize=1, stride=1, pad=0),
            Mconv1_stage4_L2=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage4_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage4_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage4_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage4_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage4_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage4_L2=L.Convolution2D(in_channels=128, out_channels=19, ksize=1, stride=1, pad=0),

            # stage5
            Mconv1_stage5_L1=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage5_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage5_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage5_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage5_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage5_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage5_L1=L.Convolution2D(in_channels=128, out_channels=38, ksize=1, stride=1, pad=0),
            Mconv1_stage5_L2=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage5_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage5_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage5_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage5_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage5_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage5_L2=L.Convolution2D(in_channels=128, out_channels=19, ksize=1, stride=1, pad=0),

            # stage6
            Mconv1_stage6_L1=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage6_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage6_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage6_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage6_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage6_L1=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage6_L1=L.Convolution2D(in_channels=128, out_channels=38, ksize=1, stride=1, pad=0),
            Mconv1_stage6_L2=L.Convolution2D(in_channels=185, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage6_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage6_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage6_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage6_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage6_L2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage6_L2=L.Convolution2D(in_channels=128, out_channels=19, ksize=1, stride=1, pad=0),
        )

    def __call__(self, x):
        heatmaps = []
        pafs = []

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_4(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3_CPM(h))
        h = F.relu(self.conv4_4_CPM(h))
        feature_map = h

        # stage1
        h1 = F.relu(self.conv5_1_CPM_L1(feature_map)) # branch1
        h1 = F.relu(self.conv5_2_CPM_L1(h1))
        h1 = F.relu(self.conv5_3_CPM_L1(h1))
        h1 = F.relu(self.conv5_4_CPM_L1(h1))
        h1 = self.conv5_5_CPM_L1(h1)
        h2 = F.relu(self.conv5_1_CPM_L2(feature_map)) # branch2
        h2 = F.relu(self.conv5_2_CPM_L2(h2))
        h2 = F.relu(self.conv5_3_CPM_L2(h2))
        h2 = F.relu(self.conv5_4_CPM_L2(h2))
        h2 = self.conv5_5_CPM_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        # stage2
        h = F.concat((h1, h2, feature_map), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage2_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage2_L1(h1))
        h1 = F.relu(self.Mconv3_stage2_L1(h1))
        h1 = F.relu(self.Mconv4_stage2_L1(h1))
        h1 = F.relu(self.Mconv5_stage2_L1(h1))
        h1 = F.relu(self.Mconv6_stage2_L1(h1))
        h1 = self.Mconv7_stage2_L1(h1)
        h2 = F.relu(self.Mconv1_stage2_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage2_L2(h2))
        h2 = F.relu(self.Mconv3_stage2_L2(h2))
        h2 = F.relu(self.Mconv4_stage2_L2(h2))
        h2 = F.relu(self.Mconv5_stage2_L2(h2))
        h2 = F.relu(self.Mconv6_stage2_L2(h2))
        h2 = self.Mconv7_stage2_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        # stage3
        h = F.concat((h1, h2, feature_map), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage3_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage3_L1(h1))
        h1 = F.relu(self.Mconv3_stage3_L1(h1))
        h1 = F.relu(self.Mconv4_stage3_L1(h1))
        h1 = F.relu(self.Mconv5_stage3_L1(h1))
        h1 = F.relu(self.Mconv6_stage3_L1(h1))
        h1 = self.Mconv7_stage3_L1(h1)
        h2 = F.relu(self.Mconv1_stage3_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage3_L2(h2))
        h2 = F.relu(self.Mconv3_stage3_L2(h2))
        h2 = F.relu(self.Mconv4_stage3_L2(h2))
        h2 = F.relu(self.Mconv5_stage3_L2(h2))
        h2 = F.relu(self.Mconv6_stage3_L2(h2))
        h2 = self.Mconv7_stage3_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        # stage4
        h = F.concat((h1, h2, feature_map), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage4_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage4_L1(h1))
        h1 = F.relu(self.Mconv3_stage4_L1(h1))
        h1 = F.relu(self.Mconv4_stage4_L1(h1))
        h1 = F.relu(self.Mconv5_stage4_L1(h1))
        h1 = F.relu(self.Mconv6_stage4_L1(h1))
        h1 = self.Mconv7_stage4_L1(h1)
        h2 = F.relu(self.Mconv1_stage4_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage4_L2(h2))
        h2 = F.relu(self.Mconv3_stage4_L2(h2))
        h2 = F.relu(self.Mconv4_stage4_L2(h2))
        h2 = F.relu(self.Mconv5_stage4_L2(h2))
        h2 = F.relu(self.Mconv6_stage4_L2(h2))
        h2 = self.Mconv7_stage4_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        # stage5
        h = F.concat((h1, h2, feature_map), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage5_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage5_L1(h1))
        h1 = F.relu(self.Mconv3_stage5_L1(h1))
        h1 = F.relu(self.Mconv4_stage5_L1(h1))
        h1 = F.relu(self.Mconv5_stage5_L1(h1))
        h1 = F.relu(self.Mconv6_stage5_L1(h1))
        h1 = self.Mconv7_stage5_L1(h1)
        h2 = F.relu(self.Mconv1_stage5_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage5_L2(h2))
        h2 = F.relu(self.Mconv3_stage5_L2(h2))
        h2 = F.relu(self.Mconv4_stage5_L2(h2))
        h2 = F.relu(self.Mconv5_stage5_L2(h2))
        h2 = F.relu(self.Mconv6_stage5_L2(h2))
        h2 = self.Mconv7_stage5_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        # stage6
        h = F.concat((h1, h2, feature_map), axis=1) # channel concat
        h1 = F.relu(self.Mconv1_stage6_L1(h)) # branch1
        h1 = F.relu(self.Mconv2_stage6_L1(h1))
        h1 = F.relu(self.Mconv3_stage6_L1(h1))
        h1 = F.relu(self.Mconv4_stage6_L1(h1))
        h1 = F.relu(self.Mconv5_stage6_L1(h1))
        h1 = F.relu(self.Mconv6_stage6_L1(h1))
        h1 = self.Mconv7_stage6_L1(h1)
        h2 = F.relu(self.Mconv1_stage6_L2(h)) # branch2
        h2 = F.relu(self.Mconv2_stage6_L2(h2))
        h2 = F.relu(self.Mconv3_stage6_L2(h2))
        h2 = F.relu(self.Mconv4_stage6_L2(h2))
        h2 = F.relu(self.Mconv5_stage6_L2(h2))
        h2 = F.relu(self.Mconv6_stage6_L2(h2))
        h2 = self.Mconv7_stage6_L2(h2)
        pafs.append(h1)
        heatmaps.append(h2)

        return pafs, heatmaps
