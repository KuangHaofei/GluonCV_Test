import os
import sys

Fast_Seg = '/home/ubuntu/workspace/gluon-cv/scripts/segmentation/Fast_Seg'
# Fast_Seg = '/Users/haofeik/workspace/cv/gluon-cv/scripts/segmentation/Fast_Seg'
sys.path.append(Fast_Seg)
print(sys.path)

import numpy as np
import torch
from torch.utils import data
import torch.nn as nn
from libs.models import ICNet, icnet
from libs.datasets.cityscapes import Cityscapes

import mxnet as mx
from mxnet import gluon, ndarray as nd
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo.icnet import get_icnet
from gluoncv.data import get_segmentation_dataset

ctx = mx.gpu()


def _AssertTensorClose(a, b, atol=1e-3, rtol=1e-3):
    npa, npb = a.cpu().detach().numpy(), b.asnumpy()
    assert np.allclose(npa, npb, atol=atol), \
        'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(
            a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())


############################# Model ###############################
# get torch weight
pt_model_path = './Fast_Seg/models/icnet_final.pth'
pt_params = torch.load(pt_model_path)
pt_model = icnet(19)
pt_model.cuda()
pt_model.eval()
pt_model.load_state_dict(pt_params, strict=False)
print('pytorch model is loaded')

# mx model
base_size = 2048
crop_size = 768
mx_model = get_icnet(nclass=19, ctx=ctx, base_size=base_size, crop_size=crop_size)
mx_model.cast('float32')
mx_model.load_parameters('./icnet_resnet50_citys.params', ctx=ctx)
print('mxnet model is loaded')


############################# Data ###############################
# pytorch DataLoader
pt_image = None
pt_label = None

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
IMG_MEAN = np.array((0, 0, 0), dtype=np.float32)
IMG_VARS = np.array((1, 1, 1), dtype=np.float32)
pt_dataset = Cityscapes('/home/ubuntu/.mxnet/datasets/citys/',
                     '/home/ubuntu/workspace/gluon-cv/scripts/segmentation/Fast_Seg/data/cityscapes/val.txt',
                     crop_size=(768, 768),
                     mean=IMG_MEAN, vars=IMG_VARS,
                     scale=False, mirror=False, RGB=False)
pt_testloader = data.DataLoader(pt_dataset, batch_size=1, shuffle=False, pin_memory=True)
for index, batch in enumerate(pt_testloader):
    pt_image, pt_label, _, _ = batch
    break

print('Pytorch Datasets:\nimage size: ', pt_image.size())
print('label size: ', pt_label.size())
############################
# mxnet DataLoader
mx_image = None
mx_label = None

base_size, crop_size = 2048, 768
# image transform
input_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    # transforms.Normalize([104.00698793, 116.66876762, 122.67891434],
    #                      [1, 1, 1]),
])
data_kwargs = {'transform': input_transform, 'base_size': base_size,
               'crop_size': crop_size}
# get dataset
mx_dataset = get_segmentation_dataset('citys', split='val', mode='val', **data_kwargs)
mx_testloader = gluon.data.DataLoader(dataset=mx_dataset, batch_size=1, last_batch='rollover', num_workers=48)

for index, batch in enumerate(mx_testloader):
    mx_image, mx_label = batch
    break

print('MXNet Datasets:\nimage size: ', mx_image.shape)
print('label size: ', mx_label.shape)

# mx_image = mx.nd.array(pt_image.cpu().data.numpy(), ctx=ctx)
_AssertTensorClose(pt_image, mx_image)
sys.exit()

############################# Compare ###############################
pt_pred = pt_model(pt_image.cuda())[0]
print(pt_pred.size())
mx_pred = mx_model(mx_image)[0]
print(mx_pred.shape)

_AssertTensorClose(pt_pred, mx_pred)
sys.exit()
