# validation with cityscapes dataset

import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import mxnet as mx
from mxnet import gluon, autograd, init
from mxnet.gluon.data.vision import transforms
from mxnet.gluon import nn

import gluoncv
from gluoncv.loss import *
from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.utils.parallel import *
from gluoncv.data import get_segmentation_dataset

from metrics import Result, AverageMeter

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon Segmentation')

    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='pascal_aug',
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default= False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    # parser.add_argument('--checkname', type=str, default='default',
    #                     help='set the checkpoint name')
    parser.add_argument('--model-zoo', type=str, default=None,
                        help='evaluating on model zoo model')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default= False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')
    # fine-tune
    parser.add_argument('--fine-tune', action='store_true', default= False,
                        help='fine-tune')
    parser.add_argument('--pretrained-dataset', type=str, default='pascal_aug',
                        help='dataset name (default: pascal)')

    # the parser
    args = parser.parse_args()

    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    print(args)
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.loss_record = []
        self.mIoU_record = []
        self.lr_record = []

        ######################### Dataset and DataLoader ###############################
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        trainset = get_segmentation_dataset(
            args.dataset, split=args.train_split, mode='train', **data_kwargs)
        valset = get_segmentation_dataset(
            args.dataset, split='val', mode='val', **data_kwargs)

        self.train_data = gluon.data.DataLoader(
            trainset, args.batch_size, shuffle=True, last_batch='rollover',
            num_workers=args.workers)
        self.eval_data = gluon.data.DataLoader(valset, args.test_batch_size,
            last_batch='rollover', num_workers=args.workers)

        ######################### Training Model ###############################
        # create network
        if args.model_zoo is not None:
            model = get_model(args.model_zoo, pretrained=True)
        else:
            if args.fine_tune:
                dataset = args.pretrained_dataset
            else:
                dataset = args.dataset

            model = get_segmentation_model(model=args.model, dataset=dataset,
                                           backbone=args.backbone, norm_layer=args.norm_layer,
                                           norm_kwargs=args.norm_kwargs, aux=args.aux,
                                           crop_size=args.crop_size)
        model.cast(args.dtype)
        # print(model)

        self.net = DataParallelModel(model, args.ctx, args.syncbn)
        self.evaluator = DataParallelModel(SegEvalModel(model), args.ctx)

        # resume checkpoint if needed
        if args.resume is not None:
            if os.path.isfile(args.resume):
                tic = time.time()
                model.load_parameters(args.resume, ctx=args.ctx)
                t_load = time.time() - tic
                print('Time of Loading Parameters: ', t_load)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'" \
                    .format(args.resume))

        # fine_tune
        if args.fine_tune:
            # TODO: Try different ways to instead the last layer of a Sequential object
            with model.name_scope():
                in_channels = model.head.block[4]._in_channels
                model.head.block._children['4'] = nn.Conv2D(in_channels=in_channels, channels=trainset.NUM_CLASS, kernel_size=1)
            model.head.block[4].initialize(init.Xavier(), ctx=args.ctx)

        ######################### Training Configuration ###############################
        # create criterion
        criterion = MixSoftmaxCrossEntropyLoss(args.aux, aux_weight=args.aux_weight)
        self.criterion = DataParallelCriterion(criterion, args.ctx, args.syncbn)

        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=args.lr,
                                        nepochs=args.epochs,
                                        iters_per_epoch=len(self.train_data),
                                        power=0.9)
        kv = mx.kv.create(args.kvstore)
        optimizer_params = {'lr_scheduler': self.lr_scheduler,
                            'wd':args.weight_decay,
                            'momentum': args.momentum,
                            'learning_rate': args.lr
                           }
        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        if args.no_wd:
            for k, v in self.net.module.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        self.optimizer = gluon.Trainer(self.net.module.collect_params(), 'sgd',
                                       optimizer_params, kvstore = kv)

        # evaluation metrics (for record final results)
        self.best_result = Result()
        self.best_result.set_to_worst()

        self.metric = gluoncv.utils.metrics.SegmentationMetric(trainset.num_class)


    def get_history(self):
        return self.loss_record, self.lr_record, self.mIoU_record


    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        count = 0
        alpha = 0.2
        for i, (data, target) in enumerate(tbar):
            with autograd.record(True):
                outputs = self.net(data.astype(args.dtype, copy=False))
                losses = self.criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.args.batch_size)
            for loss in losses:
                train_loss += np.mean(loss.asnumpy()) / len(losses)
            tbar.set_description('Epoch %d, training loss %.3f' % (epoch, train_loss/(i+1)))
            mx.nd.waitall()
            count = i
            # debug
            # if i > 5:
            #     break

        # record loss, lr
        self.loss_record.append(train_loss / (count + 1))
        self.lr_record.append(self.optimizer.learning_rate)

    def validation(self, epoch):
        #total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        avg = Result()
        result = Result()
        average_meter = AverageMeter()
        self.metric.reset()
        
        tbar = tqdm(self.eval_data)
        tic = time.time()
        for i, (data, target) in enumerate(tbar):
            data_time = time.time() - tic

            tic = time.time()
            outputs = self.evaluator(data.astype(args.dtype, copy=False))
            outputs = [x[0] for x in outputs]
            gpu_time = time.time() - tic

            targets = mx.gluon.utils.split_and_load(target, args.ctx, even_split=False)
            self.metric.update(targets, outputs)
            pixAcc, mIoU = self.metric.get()

            result.update(pixAcc, mIoU, gpu_time, data_time)
            average_meter.update(result)
            avg = average_meter.average()

            tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f' % (epoch, pixAcc, mIoU))
            mx.nd.waitall()

            tic = time.time()

            # debug
            # if i > 5:
            #     break
        self.mIoU_record.append(avg.mIoU)

        return avg


def save_checkpoint(net, args, directory, is_best=False):
    """Save Checkpoint"""
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = args.model + '_' + args.backbone + '_' + args.dataset + '_' + 'checkpoint.params'
    filename = directory + filename
    net.save_parameters(filename)
    if is_best:
        shutil.copyfile(filename, directory + args.model + '_' + args.backbone + '_' + args.dataset + '_' + 'model_best.params')


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)

    output_filename = args.model + '_' + args.backbone + '_' + args.dataset + '_'

    if args.model_zoo is not None:
        output_filename = args.model_zoo + '_'

    # Recording directory and files
    output_directory = './training/' + output_filename + '/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.eval:
        record_txt = os.path.join(output_directory, output_filename + 'eval.txt')
    else:
        record_txt = os.path.join(output_directory, output_filename + 'best.txt')

    idx = np.arange(args.epochs - args.start_epoch)

    if args.eval:
        print('Evaluating model: ', args.resume)
        result = trainer.validation(args.start_epoch)

        # record accuracy
        with open(record_txt, 'w') as txtfile:
            txtfile.write("pixAcc={:.3f}\nmIoU={:.3f}\nt_gpu={:.4f}\nt_data={:.4f}".
                          format(result.pixAcc, result.mIoU, result.gpu_time, result.data_time))
        _, _, mIoU_record = trainer.get_history()

        plt.figure()
        plt.plot(idx, mIoU_record, c='b', label='mIoU')
        plt.title('mIoU curve')
        plt.xlabel('epochs')
        plt.ylabel('mIoU')

        plt.savefig(output_directory + 'mIoU_curve_eval.png')

    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epochs:', args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            # training
            trainer.training(epoch)

            # save every epoch
            save_checkpoint(trainer.net.module, trainer.args, output_directory, False)

            # validation
            if not trainer.args.no_val:
                result = trainer.validation(epoch)

                # record best result
                is_best = result.mIoU > trainer.best_result.mIoU
                if is_best:
                    trainer.best_result = result

                    with open(record_txt, 'w') as txtfile:
                        txtfile.write("epoch={}\npixAcc={:.3f}\nmIoU={:.3f}\nt_gpu={:.4f}\nt_data={:.4f}".
                                      format(epoch, result.pixAcc, result.mIoU, result.gpu_time, result.data_time))

                    # save best model
                    save_checkpoint(trainer.net.module, trainer.args, output_directory, True)

        ########################### loss, lr, mIoU record ############################
        # save loss, lr and mIoU curve
        loss_record, lr_record, mIoU_record = trainer.get_history()

        # record file
        np.savetxt(output_directory + 'loss.txt', np.array(loss_record))
        np.savetxt(output_directory + 'lr.txt', np.array(lr_record))
        np.savetxt(output_directory + 'mIoU.txt', np.array(mIoU_record))

        # plot
        plt.figure()
        plt.plot(idx, loss_record, c='r', label='loss')
        plt.title('Loss curve')
        plt.xlabel('epochs')
        plt.ylabel('training loss')
        plt.savefig(output_directory + 'loss_curve.png')

        plt.figure()
        plt.plot(idx, lr_record, c='g', label='lr')
        plt.title('Learning rate curve')
        plt.xlabel('epochs')
        plt.ylabel('learning rate')
        plt.savefig(output_directory + 'lr_curve.png')

        plt.figure()
        plt.plot(idx, mIoU_record, c='b', label='mIoU')
        plt.title('mIoU curve')
        plt.xlabel('epochs')
        plt.ylabel('mIoU')
        plt.savefig(output_directory + 'mIoU_curve.png')

