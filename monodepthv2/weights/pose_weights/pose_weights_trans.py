import numpy as np
import torch
import torchvision
import networks

import mxnet as mx
from gluoncv.model_zoo.monodepthv2 import *

# get torch weight
pt_model_encoder_path = '../weights/mono_640x192/pose_encoder.pth'
pt_model_decoder_path = '../weights/mono_640x192/pose.pth'
pt_encoder_params = torch.load(pt_model_encoder_path, map_location='cpu')
pt_decoder_params = torch.load(pt_model_decoder_path, map_location='cpu')


# for k, v in pt_encoder_params.items():
#     print(k)
# for k, v in pt_decoder_params.items():
#     print(k)

num_layers = 18

pt_encoder = networks.ResnetEncoder(num_layers, False, 2)
pt_decoder = networks.PoseDecoder(pt_encoder.num_ch_enc, 1, 2)

pt_encoder.load_state_dict(pt_encoder_params)
pt_decoder.load_state_dict(pt_decoder_params)

pt_encoder.eval()
pt_decoder.eval()

print('pytorch model is loaded')

# mx model
context = mx.cpu()
mx_encoder = ResnetEncoder('resnet18', pretrained=False, num_input_images=2, ctx=context)
mx_decoder = PoseDecoder(mx_encoder.num_ch_enc, 1, 2)
mx_encoder.initialize(ctx=context)
mx_decoder.initialize(ctx=context)

mx_encoder.cast('float32')
mx_decoder.cast('float32')

mx_encoder_params = mx_encoder.collect_params()
mx_decoder_params = mx_decoder.collect_params()

# for k, v in mx_encoder_params.items():
#     print(k)
# for k, v in mx_decoder_params.items():
#     print(k)

######################## split weights ########################
######## encoder ########
## pytorch keys
pt_resnet_keys = []
for key in pt_encoder_params.keys():
    if 'num_batches_tracked' in key:
        continue
    pt_resnet_keys.append(key)

## mxnet keys
mx_resnet_keys = list(mx_encoder_params.keys())

# for i in range(len(mx_resnet_keys)):
#     print(pt_resnet_keys[i], '\t', "<<=============>>", '\t', mx_resnet_keys[i])

######## decoder ########

## pytorch keys
pt_decoder_keys = list(pt_decoder_params.keys())

## mxnet keys
mx_decoder_keys = list(mx_decoder_params.keys())

# for i in range(len(mx_decoder_keys)):
#     print(pt_decoder_keys[i], '\t', "<<=============>>", '\t', mx_decoder_keys[i])

######################## loading resnet18 ########################
print(len(pt_resnet_keys))
print(len(mx_resnet_keys))

for i in range(len(mx_resnet_keys)):
    mx_key_ = mx_resnet_keys[i]
    pt_key_ = pt_resnet_keys[i]
    # print('%s \t<<====================>> \t%s' % (mx_key_, pt_key_))
    mx_encoder_params[mx_key_].set_data(mx.nd.array(pt_encoder_params[pt_key_].cpu().numpy()))

print('loaded resnet weights')

######################## loading decoder ########################
print(len(pt_decoder_keys))
print(len(mx_decoder_keys))

for i in range(len(mx_decoder_keys)):
    mx_key_ = mx_decoder_keys[i]
    pt_key_ = pt_decoder_keys[i]
    # print('%s \t<<====================>> \t%s' % (mx_key_, pt_key_))
    mx_decoder_params[mx_key_].set_data(mx.nd.array(pt_decoder_params[pt_key_].cpu().numpy()))

print('loaded decoder weights')

def _AssertTensorClose(a, b, atol=1e-3, rtol=1e-3):
    npa, npb = a.cpu().detach().numpy(), b.asnumpy()
    assert np.allclose(npa, npb, atol=atol), \
        'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(
            a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())

######################## Testing ########################
# # encoder weights transfer test
pt_x = torch.rand((1, 6, 192, 640))
pt_y = pt_encoder(pt_x)

mx_x = mx.nd.array(pt_x.data.numpy())
mx_y = mx_encoder(mx_x)

for i in range(len(pt_y)):
    print(pt_y[i].shape, '\t', mx_y[i].shape)
    _AssertTensorClose(pt_y[i], mx_y[i])

# decoder weights transfer test
pt_z = pt_decoder([pt_y])
mx_z = mx_decoder([mx_y])

for i in range(len(pt_z)):
    print(pt_z[i].shape, '\t', mx_z[i].shape)
    _AssertTensorClose(pt_z[i], mx_z[i])
# exit()

######################## Save model ########################
mx_encoder.save_parameters('../weights/mono_640x192/pose_encoder.params')
mx_decoder.save_parameters('../weights/mono_640x192/pose.params')

print('model is saved')

print('to the end')
