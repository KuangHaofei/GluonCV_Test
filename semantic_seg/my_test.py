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
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete

from metrics import Result, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='Validation on Segmentation model')
    # model and dataset
    parser.add_argument('--model-zoo', type=str, default=None,
                        help='evaluating on model zoo model')
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='base network')
    parser.add_argument('--image-shape', type=int, default=480,
                        help='image shape')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--mode', type=str, default='val',
                        help='val, testval')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        help='dataset used for validation [pascal_voc, pascal_aug, coco, ade20k]')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='number of benchmarking iterations.')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--pretrained', action="store_true",
                        help='whether to use pretrained params')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')

    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default=False,
                        help='using Synchronized Cross-GPU BatchNorm')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')

    args = parser.parse_args()

    args.ctx = [mx.cpu(0)]
    args.ctx = [mx.gpu(i) for i in range(args.ngpus)] if args.ngpus > 0 else args.ctx

    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}

    print(args)
    return args


def test(model_prefix, args):
    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ######################### Dataset and DataLoader ###############################
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    # get dataset
    if args.eval:
        testset = get_segmentation_dataset(
            args.dataset, split='val', mode='testval', transform=input_transform)
    else:
        testset = get_segmentation_dataset(
            args.dataset, split='test', mode='test', transform=input_transform)

    test_data = gluon.data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, last_batch='keep',
                                      batchify_fn=ms_batchify_fn, num_workers=args.workers)

    ######################### Testing Model ###############################
    ## get the model
    if args.pretrained:
        model = get_model(model_prefix, pretrained=True)
        model.collect_params().reset_ctx(ctx=args.ctx)
        # model.hybridize()
        print("Model Type: Pre-trained model")
    else:
        # model = get_segmentation_model(model=args.model, dataset=args.dataset, ctx=args.ctx,
        #                                backbone=args.backbone, norm_layer=args.norm_layer,
        #                                norm_kwargs=args.norm_kwargs, aux=args.aux,
        #                                base_size=args.base_size, crop_size=args.crop_size)
        model = get_model(model_prefix, pretrained=False)
        model.collect_params().reset_ctx(ctx=args.ctx)

        # load local pretrained weight
        assert args.resume is not None, '=> Please provide the checkpoint using --resume'
        if os.path.isfile(args.resume):
            model.load_parameters(args.resume, ctx=args.ctx)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'" \
                               .format(args.resume))

        print('Model Type: Training from scratch')

    print("Successfully loaded %s model" % model_prefix)
    print('Testing model: ', model_prefix)

    # print(model)

    # evaluator and error metric
    evaluator = MultiEvalModel(model, testset.num_class, ctx_list=args.ctx)
    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)

    # for recording the final results
    avg = Result()
    result = Result()
    average_meter = AverageMeter()

    ######################### Testing Step ###############################
    tbar = tqdm(test_data)
    tic = time.time()
    for i, (data, dsts) in enumerate(tbar):
        data_time = time.time() - tic

        if args.eval:
            tic = time.time()
            predicts = [pred[0] for pred in evaluator.parallel_forward(data)]
            gpu_time = time.time() - tic

            targets = [target.as_in_context(predicts[0].context) \
                       for target in dsts]
            metric.update(targets, predicts)
            pixAcc, mIoU = metric.get()

            # record error metrics
            result.update(pixAcc, mIoU, gpu_time, data_time)
            average_meter.update(result)
            avg = average_meter.average()

            tbar.set_description('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            im_paths = dsts
            predicts = evaluator.parallel_forward(data)
            for predict, impath in zip(predicts, im_paths):
                predict = mx.nd.squeeze(mx.nd.argmax(predict[0], 1)).asnumpy() + \
                          testset.pred_offset
                mask = get_color_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))

        nd.waitall()
        tic = time.time()

    return avg


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.info(args)

    if args.model_zoo is not None:
        model_prefix = args.model_zoo
    else:
        model_prefix = args.model + '_' + args.backbone
        if 'pascal' in args.dataset:
            model_prefix += '_voc'
        elif args.dataset == 'coco':
            model_prefix += '_coco'
        elif args.dataset == 'ade20k':
            model_prefix += '_ade'
        elif args.dataset == 'citys':
            model_prefix += '_citys'
        else:
            raise ValueError('Unsupported dataset {} used'.format(args.dataset))

    # testing
    if args.pretrained:
        output_directory = './testing/pretrained'
    else:
        output_directory = './training/resume'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_filename = model_prefix + '_batch=' + str(args.batch_size) + '_'
    test_txt = os.path.join(output_directory, output_filename + 'test.txt')

    result = test(model_prefix, args)

    # record accuracy
    with open(test_txt, 'w') as txtfile:
        txtfile.write("pixAcc={:.3f}\nmIoU={:.3f}\nt_gpu={:.4f}\nt_data={:.4f}".
                      format(result.pixAcc, result.mIoU, result.gpu_time, result.data_time))
