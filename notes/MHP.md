## Multi-Human Parsing
### Debug Experiment 2: NaN Loss
- machine
  - mxnet-cu101mkl-1.4.1
  - gluoncv==0.7.0

- debug 1 : reduce lr and crop_size
  - experiment 0: lr = 0.001(poly); crop_size = 768
    - psp:
      - command:
        `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --checkname psp_resnet50_mhp --lr 0.001 --epochs 30 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
      - result:
        - log example: https://drive.corp.amazon.com/view/haofeik%40/train_log_example.txt
        - Epoch 12 iteration 0035/0187: training loss 0.312; learning rate 0.000624
        - Epoch 12 iteration 0036/0187: training loss nan; learning rate 0.000624
    - icnet:
      - command:
        `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 8 --checkname icnet_resnet50_mhp --lr 0.001 --epochs 30 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
      - result:
        - Epoch 9 iteration 0054/0187: training loss 0.942; learning rate 0.000716
        - Epoch 9 iteration 0055/0187: training loss nan; learning rate 0.000715
  - experiment 1: lr = 0.0005(poly); crop_size = 768
    - psp:
      - command:
        `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --checkname psp_resnet50_mhp --lr 0.0005 --epochs 30 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
      - result:
    - icnet:
      - command:
        `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 8 --checkname icnet_resnet50_mhp --lr 0.0005 --epochs 30 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
      - result:
  - experiment 2: lr = 0.001(poly); crop_size = 480
    - psp:
      - command:
        `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --checkname psp_resnet50_mhp --lr 0.001 --epochs 30 --base-size 480 --crop-size 480 --workers 48 --batch-size 16 --log-interval 1`
      - result:
  - experiment 3: lr = 0.0005(poly); crop_size = 480
    - psp:
      - command:
        `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --checkname psp_resnet50_mhp --lr 0.0005 --epochs 30 --base-size 768 --crop-size 480 --workers 48 --batch-size 16 --log-interval 1`
      - result:
### Debug Experiment 1: NaN Loss
- debug 1 : checking input samples
  - setting: lr = 0.001(constant); batch_size = 1; npgus = 1; shuffle = False
  - psp:
    - command:
      `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 1 --checkname psp_resnet50_mhp --lr 0.001 --epochs 30 --base-size 768 --crop-size 768 --workers 48 --batch-size 1 --log-interval 1`
    - experiment:
      - No nan in input samples

- debug 2 : data problem
  - setting: lr = 0.001(constant); batch_size = 1; npgus = 1; shuffle = False
  - psp:
    - command:
      `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 1 --checkname psp_resnet50_mhp --lr 0.001 --epochs 30 --base-size 768 --crop-size 768 --workers 48 --batch-size 1 --log-interval 1`
    - experiment 1:
      - Epoch 0 iteration 1705/3000: training loss nan
    - experiment 2:
      - Epoch 0 iteration 2002/3000: training loss nan
    - experiment 3:
      - Epoch 0 iteration 1596/3000: training loss nan
  - icnet:
    - command:
      `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 1 --checkname icnet_resnet50_mhp --lr 0.001 --epochs 30 --base-size 768 --crop-size 768 --workers 48 --batch-size 1 --log-interval 1`
    - hang !

- debug 3: gradient clipping
  - setting:
    - lr = 0.001(constant); batch_size = 1; npgus = 1; shuffle = False
  - psp:
    - command:
      `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 1 --checkname psp_resnet50_mhp --lr 0.001 --epochs 15 --base-size 768 --crop-size 768 --workers 48 --batch-size 1 --log-interval 1 --clip-grad 20`
    - experiment 1: --clip-grad 20
      - result: Epoch 0 iteration 1838/3000: training loss nan
    - experiment 1: --clip-grad 10
      - result: Epoch 0 iteration 1719/3000: training loss nan
    - experiment 1: --clip-grad 2
      - result: Epoch 0 iteration 1834/3000: training loss nan

### Environment Preparation
- machine : haofeik_1
- packages:
  - mxnet-cu101mkl-1.6.0b20191006
  - gluoncv==0.7.0

### Datasets Preparation
- problems:
  - gray scale
    - gray to 3 channel : baseline
    - add rgb to gray : ablation
  - resolution
    - downsample : 768
  - label
    - overlap
- Dataloader:
  - pipline:
    1. store list
        - two for-loop ?
    2. loading images
        - extend channels
    3. align masks for one input
        - boundary problem ? using masked array
    4. align resolution
        - resize directly ? yes
        - crop_size ? after resize

- Training
  - command : pspnet : haofeik_1
    `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --checkname psp_resnet50_mhp --lr 0.001 --epochs 240 --base-size 768 --crop-size 768 --workers 48`
  - command : icnet : haofeik_2
    `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 8 --checkname icnet_resnet50_mhp --lr 0.001 --epochs 240 --base-size 768 --crop-size 768 --workers 48`
