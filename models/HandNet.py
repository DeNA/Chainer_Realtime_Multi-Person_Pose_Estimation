import chainer
import chainer.functions as F
import chainer.links as L

class HandNet(chainer.Chain):
    insize = 368

    def __init__(self):
        super(HandNet, self).__init__(

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
            conv4_3=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1, pad=1),
            conv4_4=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1, pad=1),
            conv5_1=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1, pad=1),
            conv5_2=L.Convolution2D(in_channels=512, out_channels=512, ksize=3, stride=1, pad=1),
            conv5_3_CPM=L.Convolution2D(in_channels=512, out_channels=128, ksize=3, stride=1, pad=1),

            # stage1
            conv6_1_CPM=L.Convolution2D(in_channels=128, out_channels=512, ksize=1, stride=1, pad=0),
            conv6_2_CPM=L.Convolution2D(in_channels=512, out_channels=22, ksize=1, stride=1, pad=0),

            # stage2
            Mconv1_stage2=L.Convolution2D(in_channels=150, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage2=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage2=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage2=L.Convolution2D(in_channels=128, out_channels=22, ksize=1, stride=1, pad=0),

            # stage3
            Mconv1_stage3=L.Convolution2D(in_channels=150, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage3=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage3=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage3=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage3=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage3=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage3=L.Convolution2D(in_channels=128, out_channels=22, ksize=1, stride=1, pad=0),

            # stage4
            Mconv1_stage4=L.Convolution2D(in_channels=150, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage4=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage4=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage4=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage4=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage4=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage4=L.Convolution2D(in_channels=128, out_channels=22, ksize=1, stride=1, pad=0),

            # stage5
            Mconv1_stage5=L.Convolution2D(in_channels=150, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage5=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage5=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage5=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage5=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage5=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage5=L.Convolution2D(in_channels=128, out_channels=22, ksize=1, stride=1, pad=0),

            # stage6
            Mconv1_stage6=L.Convolution2D(in_channels=150, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv2_stage6=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv3_stage6=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv4_stage6=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv5_stage6=L.Convolution2D(in_channels=128, out_channels=128, ksize=7, stride=1, pad=3),
            Mconv6_stage6=L.Convolution2D(in_channels=128, out_channels=128, ksize=1, stride=1, pad=0),
            Mconv7_stage6=L.Convolution2D(in_channels=128, out_channels=22, ksize=1, stride=1, pad=0),
        )

    def __call__(self, x):
        heatmaps = []

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
        h = F.relu(self.conv4_3(h))
        h = F.relu(self.conv4_4(h))
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3_CPM(h))
        feature_map = h

        # stage1
        h = F.relu(self.conv6_1_CPM(h))
        h = self.conv6_2_CPM(h)
        heatmaps.append(h)

        # stage2
        h = F.concat((h, feature_map), axis=1) # channel concat
        h = F.relu(self.Mconv1_stage2(h))
        h = F.relu(self.Mconv2_stage2(h))
        h = F.relu(self.Mconv3_stage2(h))
        h = F.relu(self.Mconv4_stage2(h))
        h = F.relu(self.Mconv5_stage2(h))
        h = F.relu(self.Mconv6_stage2(h))
        h = self.Mconv7_stage2(h)
        heatmaps.append(h)

        # stage3
        h = F.concat((h, feature_map), axis=1) # channel concat
        h = F.relu(self.Mconv1_stage3(h))
        h = F.relu(self.Mconv2_stage3(h))
        h = F.relu(self.Mconv3_stage3(h))
        h = F.relu(self.Mconv4_stage3(h))
        h = F.relu(self.Mconv5_stage3(h))
        h = F.relu(self.Mconv6_stage3(h))
        h = self.Mconv7_stage3(h)
        heatmaps.append(h)

        # stage4
        h = F.concat((h, feature_map), axis=1) # channel concat
        h = F.relu(self.Mconv1_stage4(h))
        h = F.relu(self.Mconv2_stage4(h))
        h = F.relu(self.Mconv3_stage4(h))
        h = F.relu(self.Mconv4_stage4(h))
        h = F.relu(self.Mconv5_stage4(h))
        h = F.relu(self.Mconv6_stage4(h))
        h = self.Mconv7_stage4(h)
        heatmaps.append(h)

        # stage5
        h = F.concat((h, feature_map), axis=1) # channel concat
        h = F.relu(self.Mconv1_stage5(h))
        h = F.relu(self.Mconv2_stage5(h))
        h = F.relu(self.Mconv3_stage5(h))
        h = F.relu(self.Mconv4_stage5(h))
        h = F.relu(self.Mconv5_stage5(h))
        h = F.relu(self.Mconv6_stage5(h))
        h = self.Mconv7_stage5(h)
        heatmaps.append(h)

        # stage6
        h = F.concat((h, feature_map), axis=1) # channel concat
        h = F.relu(self.Mconv1_stage6(h))
        h = F.relu(self.Mconv2_stage6(h))
        h = F.relu(self.Mconv3_stage6(h))
        h = F.relu(self.Mconv4_stage6(h))
        h = F.relu(self.Mconv5_stage6(h))
        h = F.relu(self.Mconv6_stage6(h))
        h = self.Mconv7_stage6(h)
        heatmaps.append(h)

        return heatmaps
