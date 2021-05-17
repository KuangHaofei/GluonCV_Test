import os
import mxnet as mx
from mxnet import gluon
import gluoncv
from gluoncv.data.kitti import readlines, dict_batchify_fn

splits_dir = os.path.join(os.path.expanduser("~"), '.mxnet/datasets/kitti', 'splits')

if __name__ == '__main__':
    eval_split = "odom_9"
    assert eval_split == "odom_9" or eval_split == "odom_10", \
        "eval_split should be either odom_9 or odom_10"

    sequence_id = int(eval_split.split("_")[1])

    data_path = os.path.join(
        os.path.expanduser("~"), '.mxnet/datasets/kitti/kitti_odom')
    filenames = readlines(
        os.path.join(splits_dir, "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))

    dataset = gluoncv.data.KITTIOdomDataset(
        data_path=data_path, filenames=filenames,
        height=192, width=640, frame_idxs=[0, 1],
        num_scales=4, is_train=False, img_ext=".png")
    dataloader = gluon.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        batchify_fn=dict_batchify_fn, num_workers=12,
        pin_memory=True, last_batch='keep')
