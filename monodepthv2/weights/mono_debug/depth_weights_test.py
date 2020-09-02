import numpy as np

import mxnet as mx


if __name__ == '__main__':
    epoch0_depth_path = '../weights/mono_debug/epoch5_depth.params'
    epoch10_depth_path = '../weights/mono_debug/epoch19_depth.params'

    epoch0_depth = mx.nd.load(epoch0_depth_path)
    epoch10_depth = mx.nd.load(epoch10_depth_path)

    diff = 0

    for key in epoch0_depth.keys():
        if key == 'encoder.encoder.fc.weight' or key == 'encoder.encoder.fc.bias':
            continue

        epoch0_depth_layer = epoch0_depth[key].asnumpy()
        epoch10_depth_layer = epoch10_depth[key].asnumpy()

        layer_diff = np.linalg.norm(epoch0_depth_layer - epoch10_depth_layer)
        diff += layer_diff

    print("depth weights diff: ", diff)

    epoch0_pose_path = '../weights/mono_debug/epoch5_pose.params'
    epoch10_pose_path = '../weights/mono_debug/epoch19_pose.params'

    epoch0_pose = mx.nd.load(epoch0_pose_path)
    epoch10_pose = mx.nd.load(epoch10_pose_path)

    diff = 0
    for key in epoch0_pose.keys():
        if key == 'encoder.encoder.fc.weight' or key == 'encoder.encoder.fc.bias':
            continue

        epoch0_pose_layer = epoch0_pose[key].asnumpy()
        epoch10_pose_layer = epoch10_pose[key].asnumpy()

        layer_diff = np.linalg.norm(epoch0_pose_layer - epoch10_pose_layer)
        diff += layer_diff

    print("pose weights diff: ", diff)
