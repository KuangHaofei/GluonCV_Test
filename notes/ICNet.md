## Cityscapes

### Issue

#### machine
- Deep Learning AMI (Ubuntu 16.04) Version 26.0
- GPU instance p3.16xlarge; GPU: 64; memory: 488 GiB
- conda 4.8.1
- python: 3.6.5
- mxnet-cu101: 1.5.1; gluoncv 0.6.0
  - failure
- mxnet-cu101: 1.6.0b20191122; gluoncv 0.6.0


## ICNet

### PR

#### PR baseline
- machine: Ohio, haofeik_1; Testing
- without hybridize, use 'shape'; ceil_mod = False;
- command:
  `python train.py --dataset citys --model icnet --backbone resnet50 --syncbn --ngpus 8 --checkname icnet_resnet50_citys --lr 0.01 --epochs 240 --base-size 2048 --crop-size 768 --workers 48`
- Training : 8 * GPU; crop_size = 768; test mode = 'val'
  - pixAcc: 95.5
  - mIoU: 67.8
- Testing : 1 * GPU; test mode = 'testval'
  - using test_icnet.py
  - pixAcc: 95.5
  - mIoU: 74.8
  - t_gpu: 36.91ms; 27 fps
- Testing : 1 * GPU; test mode = 'testval'
  - command:
    `python test.py --dataset citys --model icnet --backbone resnet50 --syncbn --ngpus 1 --base-size 2048 --workers 48 --eval --resume ./runs/citys/icnet/icnet_resnet50_citys/`
  - pixAcc: 95.5
  - mIoU: 74.8
  - t_gpu: 36.91ms; 27 fps

### MXNet Compare Experiments

#### Resnet Arch Compare
- dilation same as Pytorch: Ohio, haofeik_3
  - pixAcc:
  - mIoU:
  - t_gpu:  fps

- dilation using original gluoncv Implementation
  - pixAcc: 95.7
  - mIoU: 74.6
  - t_gpu: 28.82 fps

### Hybridize Debug
#### Debug 2
- machine: Ohio, haofeik_2
- Full Hybridize
- Training : 8 * GPU; crop_size = 768; test mode = 'val'
  - pixAcc:
  - mIoU:
- Testing : 1 * GPU; test mode = 'testval'
  - pixAcc:
  - mIoU:
  - t_gpu: fps

#### Debug 1
- machine: Ohio, haofeik_1
- no Hybridize
- Training : 8 * GPU; crop_size = 768; test mode = 'val'
  - pixAcc: 95.5
  - mIoU: 67.7
- Testing : 1 * GPU; test mode = 'testval'
  - pixAcc: 93.5
  - mIoU: 67.8
  - t_gpu: 38.46ms; 26 fps

#### Debug baseline
- machine: Ohio, haofeik; Finised
- without hybridize, use 'shape'
- Training : 8 * GPU; crop_size = 768; test mode = 'val'
  - pixAcc: 95.5
  - mIoU: 68.2
- Testing : 1 * GPU; test mode = 'testval'
  - pixAcc: 95.7
  - mIoU: 75.6
  - t_gpu: 36.17ms; 28 fps

#### hybridize bug
- psp:
  ```
  File "/home/ubuntu/workspace/gluon-cv/gluoncv/model_zoo/pspnet.py", line 52, in hybrid_forward
    c3, c4 = self.base_forward(x)
  File "/home/ubuntu/workspace/gluon-cv/gluoncv/model_zoo/segbase.py", line 77, in base_forward
    x = self.conv1(x)
  ```
- fcn:
  ```
  File "/home/ubuntu/workspace/gluon-cv/gluoncv/model_zoo/fcn.py", line 57, in hybrid_forward
    c3, c4 = self.base_forward(x)
  File "/home/ubuntu/workspace/gluon-cv/gluoncv/model_zoo/segbase.py", line 82, in base_forward
    x = self.layer2(x)
  ......
  File "/home/ubuntu/workspace/gluon-cv/gluoncv/model_zoo/resnetv1b.py", line 91, in hybrid_forward
  out = self.relu1(out)
  ```
- deeplab:
  ```
  File "/home/ubuntu/workspace/gluon-cv/gluoncv/model_zoo/segbase.py", line 77, in base_forward
    x = self.conv1(x)
  ```


### MXNet Training & Testing Experiments

#### Change to hybridize
- Training : 8 * GPU; crop_size = 768; test mode = 'val'
  - pixAcc: 95.4
  - mIoU: 66.8
- Testing : 1 * GPU; test mode = 'testval'
  - pixAcc: 95.4
  - mIoU: 72.6
  - t_gpu: 37.20ms; 27 fps

#### Train 2
- machine: Ohio, haofeik_2
- Arch: use original gluoncv
- Setting:
  - lr: 0.01; poly with power 0.9
  - momentum: 0.9
  - weight decay: 0.0001
  - auxiliary loss: (0.4, 0.4, 1.0)
  - data augmentation
  - epochs: 240
- Training : 8 * GPU; crop_size = 768; test mode = 'val'
  - pixAcc: 95.5
  - mIoU: 67.3
- Testing : 1 * GPU; test mode = 'testval'
  - pixAcc: 95.7
  - mIoU: 74.6
  - t_gpu: 29 fps

#### Train 1
- machine: Ohio, haofeik_1
- Arch : same as pytorch icnet
- Setting:
  - lr: 0.01; poly with power 0.9
  - momentum: 0.9
  - weight decay: 0.0001
  - auxiliary loss: (0.4, 0.4, 1.0)
  - data augmentation
  - epochs: 240
- Training : 8 * GPU; crop_size = 768; test mode = 'val'
  - pixAcc: 95.4
  - mIoU: 66.1
- Testing : 1 * GPU; test mode = 'testval'
  - pixAcc: 95.7
  - mIoU: 74.6
  - t_gpu: 28 fps

### Pytorch Implementation
https://github.com/lxtGH/Fast_Seg

#### ICNet Pretrained model
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=1A6z87_GCHEuKeZfbGpEvnkZ0POdW2Q_U" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1A6z87_GCHEuKeZfbGpEvnkZ0POdW2Q_U" -O icnet_final.pth
```

#### ICNet Validation
```
python val.py --data_dir ~/.mxnet/datasets/citys/ --data_list ./data/cityscapes/val.txt --arch icnet --restore_from ./models/icnet_final.pth --gpu 8 --whole True
```

## Model Validation
- pspnet; citys:
  - scripts
  ```
  python ./scripts/segmentation/train.py --dataset citys --model psp --aux --backbone resnet101 --syncbn --ngpus 8 --checkname psp_resnet101_citys --lr 0.01 --epochs 240 --base-size 2048 --crop-size 768 --workers 48
  ```
  ```
  python ./scripts/segmentation/test.py --model psp --backbone resnet101 --dataset citys --batch-size 8 --ngpus 8 --eval --resume /home/ubuntu/workspace/cv/GluonCV_Test/semantic_seg/training/psp_resnet101_citys_/psp_resnet101_citys_model_best.params --aux --base-size 2048 --crop-size 768
  ```
