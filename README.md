# GluonCV_Test

## ICNet

### MXNet Compare Experiments

#### Resnet Arch Compare
- dilation same as Pytorch
  - pixAcc:
  - mIoU:
  - t_gpu:  fps

- dilation using original gluoncv Implementation
  - pixAcc: 95.7
  - mIoU: 74.6
  - t_gpu: 28.82 fps

### MXNet Training & Testing Experiments

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
  - t_gpu: 28.82 fps

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
  - t_gpu: 27.85 fps

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


## AWS Initialize
- Login
```
ssh -i haofeik.pem ubuntu@3.137.3.240
source activate mxnet_p36
```
```
source activate mxnet_p36
sudo apt-get install -y pandoc
pip install --upgrade pip
pip install --upgrade --pre mxnet-cu101mkl
pip install Cython
pip install pypandoc pycocotools
cd ~/
git clone https://github.com/hetong007/gluon-cv.git
cd gluon-cv
pip install -e .
pip install opencv-python
git config --global user.name "Kuang Haofei"
git config --global user.email "haofeikuang@gmail.com"
```

- Mount
```
sudo sshfs -C -o reconnect ubuntu@3.137.3.240:/home/ubuntu/workspace ~/Desktop/remote/ -o IdentityFile=~/haofeikuang.pem -o allow_other
```

- Jupyter Notebook
```
# aws
jupyter notebook --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key
```
```
# client
ssh -i haofeikuang.pem -N -f -L 8888:localhost:8888 ubuntu@3.137.3.240
```

## Datasets Preparation
- Cityscape
  - download filw
  ```
  https://gluon-cv.mxnet.io/build/examples_datasets/cityscapes.html
  ```
  - Save login cookies
  ```
  wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=kuanghf@shanghaitech.edu.cn&password=4eba584a&submit=Login' https://www.cityscapes-dataset.com/login/; history -d $((HISTCMD-1))
  ```
  - downloading
  ```
  wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=PACKAGE_ID
  ```

## Git PR
```
git status  # forked
git remote -v
git remote add upstream https://github.com/dmlc/gluon-cv.git

git remote -v
git fetch upstream
git merge upstream/master
git push

git checkout -b fix_segmentation_test   # new branch for PR
# modify something
git add test.py
git status
git commit -m "fixed test.py"
git push -u origin fix_segmentation_test

# if don't setup username
git config --global user.name "Kuang Haofei"
git config --global user.email "haofeikuang@gmail.com"
git commit --amend
git pull
git push

# In forked webpage, add New Pull Request
```
