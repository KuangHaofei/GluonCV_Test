import mxnet as mx
import gluoncv

import numpy as np

class Result(object):
    def __init__(self):
        self.pixAcc = 0.0
        self.mIoU = 0.0

        self.gpu_time = 0.0
        self.data_time = 0.0

    def set_to_worst(self):
        self.pixAcc = 0.0
        self.mIoU = 0.0
        self.data_time, self.gpu_time = 0.0, 0.0

    def update(self, pixAcc, mIoU, gpu_time=0.0, data_time=0.0):
        self.pixAcc = pixAcc
        self.mIoU = mIoU

        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        pass

class AverageMeter(object):
    def __init__(self):
        self.count = 0.0
        self.pixAcc = 0.0
        self.mIoU = 0.0

        self.sum_gpu_time = 0.0
        self.sum_data_time = 0.0

    def update(self, result, n=1):
        self.count += n

        self.pixAcc = result.pixAcc
        self.mIoU = result.mIoU

        self.sum_gpu_time += n * result.gpu_time
        self.sum_data_time += n * result.data_time

    def average(self):
        avg = Result()
        avg.update(self.pixAcc, self.mIoU,
                   self.sum_gpu_time / self.count, self.sum_data_time / self.count)

        return avg