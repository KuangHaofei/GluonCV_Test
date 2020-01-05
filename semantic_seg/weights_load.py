import numpy as np
import torch
import torchvision
from Fast_Seg.libs.models import ICNet, icnet

import mxnet as mx
from mxnet import nd
import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo.icnet import get_icnet

import sys

# get torch weight
pt_model_path = '/Users/haofeik/workspace/cv/Fast_Seg/models/icnet_final.pth'
pt_params = torch.load(pt_model_path, map_location='cpu')
pt_model = icnet(19)
pt_model.eval()
pt_model.load_state_dict(pt_params, strict=False)

print('pytorch model is loaded')

# mx model
crop_size = 480
context = mx.cpu()
mx_model = get_icnet(nclass=19, crop_size=crop_size)
mx_model.cast('float32')
mx_params = mx_model.collect_params()

######################## split weights ########################
## pytorch keys
pt_resnet_keys = []
pt_psphead_keys = []
pt_cff_keys = []
pt_conv_sub_keys = []

for key in pt_params.keys():
    if 'num_batches_tracked' in key:
        continue

    if key.startswith('backbone'):
        if 'head' in key:
            # pspnet head
            pt_psphead_keys.append(key)
        else:
            # resnet50
            pt_resnet_keys.append(key)
    elif key.startswith('head'):
        # cff
        pt_cff_keys.append(key)
    elif key.startswith('conv_sub'):
        # conv_sub
        pt_conv_sub_keys.append(key)

## mxnet keys
mx_resnet_keys = []
mx_psphead_keys = []
mx_cff_keys = []
mx_conv_sub_keys = []

for key in mx_params.keys():
    if key.startswith('icnet0_resnetv1s'):
        # resnet50
        mx_resnet_keys.append(key)
    elif key.startswith('icnet0__p'):
        # pspnet head
        mx_psphead_keys.append((key))
    # cff
    elif key.startswith('icnet0_cascadefeaturefusion'):
        mx_cff_keys.append(key)
    elif key == 'icnet0__ichead0_conv0_weight':
        mx_cff_keys.append(key)
    # conv_sub
    else:
        mx_conv_sub_keys.append(key)


######################## loading resnet50 ########################
print(len(pt_resnet_keys))
print(len(mx_resnet_keys))

for i in range(len(pt_resnet_keys)):
    mx_key_ = mx_resnet_keys[i]
    pt_key_ = pt_resnet_keys[i]
    # print('%s \t<<====================>> \t%s' % (mx_key_, pt_key_))
    mx_params[mx_key_].set_data(nd.array(pt_params[pt_key_].cpu().numpy()))

print('loaded resnet weights')
# sys.exit()

######################## loading psphead ########################
print(len(pt_psphead_keys))
print(len(mx_psphead_keys))

for i in range(len(mx_psphead_keys)):
    mx_key_ = mx_psphead_keys[i]
    pt_key_ = pt_psphead_keys[i]
    # print('%s \t<<====================>> \t%s' % (mx_key_, pt_key_))
    mx_params[mx_key_].set_data(nd.array(pt_params[pt_key_].cpu().numpy()))

print('loaded psphead weights')

######################## loading cff ########################
print(len(pt_cff_keys))
print(len(mx_cff_keys))

for i in range(len(pt_cff_keys)):
    mx_key_ = mx_cff_keys[i]
    pt_key_ = pt_cff_keys[i]
    # print('%s \t<<====================>> \t%s' % (mx_key_, pt_key_))
    mx_params[mx_key_].set_data(nd.array(pt_params[pt_key_].cpu().numpy()))

print('loaded cff weights')

######################## loading conv_sub ########################
print(len(pt_conv_sub_keys))
print(len(mx_conv_sub_keys))

for i in range(len(pt_conv_sub_keys)):
    mx_key_ = mx_conv_sub_keys[i]
    pt_key_ = pt_conv_sub_keys[i]
    # print('%s \t<<====================>> \t%s' % (mx_key_, pt_key_))
    mx_params[mx_key_].set_data(nd.array(pt_params[pt_key_].cpu().numpy()))

print('loaded conv_sub weights')

# testing
# tmp_input = mx.ndarray.ones((1,3,480,480))
# mx_model.summary(tmp_input.as_in_context(context))
# sys.exit()


def _AssertTensorClose(a, b, atol=1e-3, rtol=1e-3):
    npa, npb = a.cpu().detach().numpy(), b.asnumpy()
    assert np.allclose(npa, npb, atol=atol), \
        'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(
            a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())


tx = torch.rand((1, 3, 480, 480))
ty = pt_model(tx)[0]
print(ty.size())

mx = mx.nd.array(tx.data.numpy())
my = mx_model(mx)[0]
print(my.shape)

_AssertTensorClose(ty, my)
sys.exit()

mx_model.save_parameters('./icnet_resnet50_citys.params')

print('model is saved')

print('to the end')
sys.exit()
