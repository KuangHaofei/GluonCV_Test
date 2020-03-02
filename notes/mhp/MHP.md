# Multi-Human Parsing Datasets

## Multi-Human Parsing v2


## Multi-Human Parsing v1

## PR 1
- doc 1 : datasets preparation
  - downloader time ?

- doc 2 : demo
  - pretrained model

- Requirements
  ```
  conda create --name mhp python=3.6
  source activate mhp

  pip install mxnet-cu101mkl==1.4.1
  pip install Cython
  pip install pypandoc pycocotools

  git clone https://github.com/dmlc/gluon-cv
  cd gluon-cv
  pip install -e .
  ```

- Dataset Preparation: mhp_v1.py
  - requirement:
      `pip install html5lib googleDriveFileDownloader`
  - usage:
    ```
    cd ~/gluoncv/scripts/dataset
    python mhp_v1.py
    ```

- Dataloader

- Training Step:
  - params: icnet_resnet50; adam; lr = 0.00001; crop_size = 768; epochs = 120;
  - command:
    - icnet:
      `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 8 --lr 0.00001 --epochs 120 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
    - results:
      - mIoU: 39.74 (Epoch 105)
      - NaN: epoch 10 ~ epoch 23
        - Epoch 10 iteration 0113/0188: training loss 1.059
        - Epoch 10 iteration 0114/0188: training loss nan
      - training log: https://github.com/KuangHaofei/GluonCV_Test/blob/master/notes/mhp_pr/train.log

- Evaluations:
  - weights: epoch_0105_mIoU_0.3974.params
  - command:
    `python test.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 1 --workers 48 --eval --resume ./runs/mhp/icnet/resnet50/epoch_0105_mIoU_0.3974.params`
  - results:
    - pixAcc: 89.52
    - mIoU: 40.41
    - t_gpu: 49.91ms

- Visualization:
  https://github.com/KuangHaofei/gluon-cv/blob/multi-human-parsing/scripts/segmentation/visualization_demo.ipynb


### Experiment : Adam optimizer
- parameters:
  - crop_size = 768
  - epochs = 120

- case 1: adam; lr = 0.0005; crop_size = 768; epochs = 120
  - command:
    - pspnet:
      `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --checkname psp_resnet50_mhp --lr 0.0005 --epochs 120 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
    - icnet:
      `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 8 --checkname icnet_resnet50_mhp --lr 0.0005 --epochs 120 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
  - results:
    - mIoU:
      - psp: 19.36 (Epoch 0)
      - icnet : 27.58 (Epoch 1)
    - NaN:
      - psp:
        - Epoch 1 iteration 0019/0187: training loss 0.700
        - Epoch 1 iteration 0020/0187: training loss nan
      - icnet:
        - Epoch 0 iteration 0024/0187: training loss 3.885
        - Epoch 0 iteration 0025/0187: training loss nan

- case 2: adam; lr = 0.0001; crop_size = 768; epochs = 120
  - command:
    - pspnet:
      `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --checkname psp_resnet50_mhp --lr 0.0001 --epochs 120 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
    - icnet:
      `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 8 --checkname icnet_resnet50_mhp --lr 0.0001 --epochs 120 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
  - results:
    - mIoU:
      - psp: 33.45 (Epoch 2)
      - icnet: 35.12 (Epoch 118)
    - NaN:
      - psp:
        - Epoch 3 iteration 0020/0187: training loss 0.466
        - Epoch 3 iteration 0021/0187: training loss nan
      - icnet:
        - Epoch 1 iteration 0051/0187: training loss 1.428
        - Epoch 1 iteration 0052/0187: training loss nan


### Visualization
- task1 : color palette
  https://gluon-cv.mxnet.io/_modules/gluoncv/utils/viz/segmentation.html#get_color_pallete

- task2 : 10 images Visualization
  https://gluon-cv.mxnet.io/build/examples_segmentation/demo_psp.html

- MHP Visualization Demo
  - model: psp_resnet50_mhp
  - weights: epoch_0042_mIoU_0.3424.params
    - lr = 0.0005
    - crop_size = 480
    - sgd
  - result:
    https://github.com/KuangHaofei/gluon-cv/blob/multi-human-parsing/scripts/segmentation/visualization_demo.ipynb

### Debug Experiment 3
- machine
  - mxnet-cu101mkl-1.4.1
  - gluoncv==0.7.0

- parameters:
  - lr = 0.0005(poly)
  - crop_size = 480

- pspnet:
  - train:
    - command:
      `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --checkname psp_resnet50_mhp --lr 0.0005 --epochs 240 --base-size 768 --crop-size 480 --workers 48 --batch-size 16 --log-interval 1`
    - result:
      - NaN:
        - Epoch 54 iteration 0033/0187: training loss 0.560
        - Epoch 54 iteration 0034/0187: training loss nan
      - best results: Epoch 46
        - pixAcc: 0.848
        - mIoU: 0.342
        - loss: Epoch 46 : 0.522
        - model:
  - test: multi-scale
    - model: epoch_0046_mIoU_0.3416.params
    - command:
      `python test.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --batch-size 8 --base-size 768 --workers 48 --eval --resume ./runs/mhp/psp/resnet50/epoch_0046_mIoU_0.3416.params`
    - result:
      - pixAcc: 0.8658
      - mIoU: 0.3061

- icnet:
  - train:
    - command:
      `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 8 --checkname icnet_resnet50_mhp --lr 0.0005 --epochs 240 --base-size 768 --crop-size 480 --workers 48 --batch-size 16 --log-interval 1`
    - result:
      - NaN:
        - Epoch 46 iteration 0079/0187: training loss 2.472
        - Epoch 46 iteration 0080/0187: training loss nan
      - best results: Epoch 24
        - pixAcc: 0.842
        - mIoU: 0.309
        - loss: 1.50
        - model:
  - test:
    - command:
      `python test.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 1 --batch-size 1 --base-size 768 --workers 48 --eval --resume ./runs/mhp/icnet/resnet50/epoch_0024_mIoU_0.3095.params`
    - result: single-scale
      - pixAcc:
      - mIoU:

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
      - result: finished
        - log: https://github.com/KuangHaofei/GluonCV_Test/blob/master/notes/mhp_example/train_psp.log
        - NaN: Epoch 18 iteration 0089/0187: training loss nan; learning rate 0.000210; But NaN disappeared in the next epoch!
        - loss: 0.245
        - mIoU: 20.5
    - icnet:
      - command:
        `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 8 --checkname icnet_resnet50_mhp --lr 0.0005 --epochs 30 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
      - result: finished
        - log: https://github.com/KuangHaofei/GluonCV_Test/blob/master/notes/mhp_example/train_icnet.log
        - loss: 0.731
        - mIoU: 17.1

  - experiment 2: lr = 0.001(poly); crop_size = 480
    - psp:
      - command:
        `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --checkname psp_resnet50_mhp --lr 0.001 --epochs 30 --base-size 480 --crop-size 480 --workers 48 --batch-size 16 --log-interval 1`
      - result: failed
        - Epoch 14 iteration 0031/0187: training loss 0.323; learning rate 0.000561
        - Epoch 14 iteration 0032/0187: training loss nan; learning rate 0.000561
        - mIoU: 16.5
    - icnet:
      - command:
        `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 8 --checkname icnet_resnet50_mhp --lr 0.001 --epochs 30 --base-size 768 --crop-size 480 --workers 48 --batch-size 16 --log-interval 1`
      - result: failed
        - Epoch 12 iteration 0048/0187: training loss 1.602; learning rate 0.000622
        - Epoch 12 iteration 0049/0187: training loss nan; learning rate 0.000622
        - mIoU: 11.8

  - experiment 3: lr = 0.0005(poly); crop_size = 480
    - psp:
      - command:
        `python train.py --dataset mhp --model psp --backbone resnet50 --syncbn --ngpus 8 --checkname psp_resnet50_mhp --lr 0.0005 --epochs 30 --base-size 768 --crop-size 480 --workers 48 --batch-size 16 --log-interval 1`
      - result: finished
        - loss: 0.355
        - mIoU: 19.0

    - icnet:
      - command:
        `python train.py --dataset mhp --model icnet --backbone resnet50 --syncbn --ngpus 8 --checkname icnet_resnet50_mhp --lr 0.0005 --epochs 30 --base-size 768 --crop-size 480 --workers 48 --batch-size 16 --log-interval 1`
      - result: finished
        - loss: 0.991
        - mIoU: 17.0


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
