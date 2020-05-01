# Multi-Human Parsing Datasets

## Multi-Human Parsing v2

### Exp : 1
- Training Step:
  - params: icnet_resnet50; adam; lr = 0.00001; crop_size = 768; epochs = 120;
  - command:
    - icnet:
      `python train.py --dataset mhpv2 --model icnet --backbone resnet50 --syncbn --ngpus 8 --optimizer adam --lr 0.00001 --epochs 120 --base-size 768 --crop-size 768 --workers 48 --batch-size 16 --log-interval 1`
    - results:
      - mIoU:
      - training log:

- Evaluations:
  - weights:
  - command:
    `python test.py --dataset mhpv2 --model icnet --backbone resnet50 --syncbn --ngpus 1 --workers 48 --eval --resume ./runs/mhp/icnet/resnet50/epoch_0105_mIoU_0.3974.params`
  - results:
    - pixAcc:
    - mIoU:
    - t_gpu:

### Experiment Setting
- machine : haofeik_

- Requirements
  ```
  conda create --name mhp python=3.6
  source activate mhp

  pip install mxnet-cu101mkl==1.4.1
  pip install Cython
  pip install pypandoc pycocotools

  git clone https://github.com/KuangHaofei/gluon-cv
  cd gluon-cv
  pip install -e .
  ```

### Dataset
- download:
```
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?id=1YVBGMru0dlwB8zu1OoErOazZoc8ISSJn&export=download" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YVBGMru0dlwB8zu1OoErOazZoc8ISSJn" -O LV-MHP-v2.zip
```

- categories: 58 + background

- total: 25403

- training: 15403 + 5000

- testing: 5000
