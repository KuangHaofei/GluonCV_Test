import numpy as np
import torch
import torchvision
import networks

import mxnet as mx
from mxnet import nd
from gluoncv.model_zoo import monodepthv2


# get torch weight
pt_model_encoder_path = './mono+stereo_640x192/encoder.pth'
pt_model_decoder_path = './mono+stereo_640x192/depth.pth'
pt_encoder_params = torch.load(pt_model_encoder_path, map_location='cpu')
pt_decoder_params = torch.load(pt_model_decoder_path, map_location='cpu')

# for k, v in pt_encoder_params.items():
#     print(k)
# for k, v in pt_decoder_params.items():
#     print(k)

num_layers = 18

pt_encoder = networks.ResnetEncoder(num_layers, False)
pt_decoder = networks.DepthDecoder(pt_encoder.num_ch_enc)

model_dict = pt_encoder.state_dict()
pt_encoder.load_state_dict({k: v for k, v in pt_encoder_params.items() if k in model_dict})
pt_decoder.load_state_dict(pt_decoder_params)

pt_encoder.eval()

print('pytorch model is loaded')

# mx model
context = mx.cpu()
mx_encoder = monodepthv2.ResnetEncoder(num_layers, pretrained=False, ctx=context)
mx_decoder = monodepthv2.DepthDecoder(mx_encoder.num_ch_enc)
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
    # print(pt_resnet_keys[i], '\t', mx_resnet_keys[i])
# exit()

######## decoder ########

## pytorch keys
pt_decoder_keys = list(pt_decoder_params.keys())

## mxnet keys
mx_decoder_keys = list(mx_decoder_params.keys())

# for i in range(len(mx_decoder_keys)):
#     print(pt_decoder_keys[i], '\t', mx_decoder_keys[i])
# exit()

######################## loading resnet18 ########################
print(len(pt_resnet_keys))
print(len(mx_resnet_keys))

for i in range(len(mx_resnet_keys)):
    mx_key_ = mx_resnet_keys[i]
    pt_key_ = pt_resnet_keys[i]
    # print('%s \t<<====================>> \t%s' % (mx_key_, pt_key_))
    mx_encoder_params[mx_key_].set_data(nd.array(pt_encoder_params[pt_key_].cpu().numpy()))

print('loaded resnet weights')

######################## loading decoder ########################
print(len(pt_decoder_keys))
print(len(mx_decoder_keys))

for i in range(len(mx_decoder_keys)):
    mx_key_ = mx_decoder_keys[i]
    pt_key_ = pt_decoder_keys[i]
    # print('%s \t<<====================>> \t%s' % (mx_key_, pt_key_))
    mx_decoder_params[mx_key_].set_data(nd.array(pt_decoder_params[pt_key_].cpu().numpy()))

print('loaded decoder weights')

# testing
# tmp_input = mx.ndarray.ones((16,3,192,640))
# mx_encoder.summary(tmp_input.as_in_context(context))
# exit()

def _AssertTensorClose(a, b, atol=1e-3, rtol=1e-3):
    npa, npb = a.cpu().detach().numpy(), b.asnumpy()
    assert np.allclose(npa, npb, atol=atol), \
        'Tensor close check failed\n{}\n{}\nadiff={}, rdiff={}'.format(
            a, b, np.abs(npa - npb).max(), np.abs((npa - npb) / np.fmax(npa, 1e-5)).max())

######################## Testing ########################
# # encoder weights transfer test
tx = torch.rand((16, 3, 192, 640))
ty = pt_encoder(tx)

mx = mx.nd.array(tx.data.numpy())
my = mx_encoder(mx)
# for i in range(len(ty)):
#     print(ty[i].shape, '\t', my[i].shape)
#     _AssertTensorClose(ty[i], my[i])
#
# # decoder weights transfer test
tz = pt_decoder(ty)
mz = mx_decoder(my)
#
# for key in tz.keys():
#     print(tz[key].shape, '\t', mz[key].shape)
#     _AssertTensorClose(tz[key], mz[key])

######################## Save model ########################
mx_encoder.save_parameters('./mx_mono+stereo_640x192/encoder.params')
mx_decoder.save_parameters('./mx_mono+stereo_640x192/depth.params')

print('model is saved')

print('to the end')