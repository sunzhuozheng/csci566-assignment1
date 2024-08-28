from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########

            ConvLayer2D(3, 3, 3),
            MaxPoolingLayer(2, 2, "Maxpool"),
            flatten(),
            fc(27, 5, init_scale=0.02)

            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########

            ConvLayer2D(3, 5, 6, stride=1, padding=2, name="conv1"),
            MaxPoolingLayer(2, 2, "Maxpool1"),
            ConvLayer2D(6, 5, 16, name="conv2"),
            MaxPoolingLayer(2, 2, "Maxpool2"),
            flatten(),
            fc(576, 10)

            ########### END ###########
        )
