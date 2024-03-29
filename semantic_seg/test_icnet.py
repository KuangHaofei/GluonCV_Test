import os
import logging
from tqdm import tqdm
import numpy as np
import argparse
import time
import sys

import mxnet as mx
from mxnet import gluon, ndarray as nd
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import *

import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo.icnet import get_icnet_resnet50_citys
from gluoncv.data import get_segmentation_dataset

from gluoncv.utils.parallel import *


def test(model_prefix):
    ngpus = 1
    syncbn = True

    ctx = [mx.cpu(0)]
    ctx = [mx.gpu(i) for i in range(ngpus)] if ngpus > 0 else ctx

    norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if syncbn else mx.gluon.nn.BatchNorm
    norm_kwargs = {'num_devices': ngpus} if syncbn else {}

    batch_size = 1
    num_workers = 48

    base_size = 2048
    height, width = 1024, 2048
    crop_size = 768

    ######################### Dataset and DataLoader ###############################
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    # ???
    # data_kwargs = {'transform': input_transform, 'base_size': base_size,
    #                'crop_size': crop_size}

    # get dataset
    valset = get_segmentation_dataset('citys', split='val', mode='testval', transform=input_transform)
    eval_data = gluon.data.DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, last_batch='rollover', num_workers=num_workers)

    ######################### Testing Model ###############################
    ## get the model
    model = get_icnet_resnet50_citys(pretrained=False, ctx=ctx,
                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                     base_size=base_size, height=height, width=width)
    model.collect_params().reset_ctx(ctx=ctx)

    resume_path = '/home/ubuntu/workspace/gluon-cv/scripts/segmentation/runs/citys/icnet/icnet_resnet50_citys/checkpoint.params'
    model.load_parameters(resume_path, ctx=ctx)

    print("Successfully loaded %s model" % model_prefix)
    print('Testing model: ', model_prefix)

    # print(model)

    # evaluator and error metric
    evaluator = DataParallelModel(SegEvalModel(model), ctx_list=ctx)
    metric = gluoncv.utils.metrics.SegmentationMetric(valset.num_class)

    ######################### Testing Step ###############################
    tbar = tqdm(eval_data)
    pixAcc, mIoU, t_gpu = 0, 0, 0
    num = 0
    for i, (data, dsts) in enumerate(tbar):
        tic = time.time()
        outputs = evaluator(data.astype('float32', copy=False))
        t_gpu += time.time() - tic
        num += 1

        outputs = [x[0] for x in outputs]
        targets = mx.gluon.utils.split_and_load(dsts, ctx_list=ctx, even_split=False)
        metric.update(targets, outputs)

        pixAcc, mIoU = metric.get()
        tbar.set_description('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))

        nd.waitall()
    t_gpu /= num

    return pixAcc, mIoU, t_gpu


if __name__ == "__main__":
    # testing
    model_prefix = 'icnet_psp50_citys_'
    pixAcc, mIoU, t_gpu = test(model_prefix)

    # testing performance
    output_directory = './testing/icnet_psp50/'
    output_filename = model_prefix

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    test_txt = os.path.join(output_directory, output_filename + 'test.txt')

    # record accuracy
    with open(test_txt, 'w') as txtfile:
        txtfile.write("pixAcc={:.3f}\nmIoU={:.3f}\nt_gpu={:.2f}ms".
                      format(pixAcc, mIoU, t_gpu*1000))
