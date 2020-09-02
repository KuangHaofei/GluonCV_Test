import os
import numpy as np
import mxnet as mx
from gluoncv.model_zoo.monodepthv2 import *

# mx model
context = mx.cpu()

mx_posenet = get_monodepth2_resnet18_posenet_kitti_mono_640x192(
    pretrained_base=True, num_input_images=2,
    num_input_features=1, num_frames_to_predict_for=2)
mx_posenet.cast('float32')

mx_posenet_params = mx_posenet.collect_params()

for k, v in mx_posenet_params.items():
    print(k)
