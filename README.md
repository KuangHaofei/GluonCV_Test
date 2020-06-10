# GluonCV_Test

## AWS Initialize
- Login
```
ssh -i haofeik.pem ubuntu@52.22.213.132
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
git clone https://github.com/KuangHaofei/gluon-cv
cd gluon-cv
pip install -e .
pip install opencv-python
git config --global user.name "Kuang Haofei"
git config --global user.email "haofeikuang@gmail.com"
```

## AWS Operations
- Generate public key form .pem file
```
openssl rsa -pubout -in haofeik.pem
```

- Mount
```
sudo sshfs -C -o reconnect ubuntu@52.22.213.132:/home/ubuntu/workspace ~/Desktop/remote/ -o IdentityFile=~/haofeikuang.pem -o allow_other
```

- Jupyter Notebook
```
# aws
jupyter notebook --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key
```
```
# client
ssh -i haofeik.pem -N -f -L 8888:localhost:8888 ubuntu@52.22.213.132
```
```
# problems:
# The 'kernel_spec_manager_class' trait of <notebook.notebookapp.NotebookApp object
# at 0x7f1799bd5590> instance must be a type, but
# 'environment_kernels.EnvironmentKernelSpecManager' could not be imported

pip install environment_kernels
```

## Datasets Preparation
- Cityscape
  - download file
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
- Multi-Human Parsing
  - MHP-v1
  ```
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?id=1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5&export=download" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5" -O LV-MHP-v1.zip
  ```
  - MHP-v2
  ```
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?id=1YVBGMru0dlwB8zu1OoErOazZoc8ISSJn&export=download" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YVBGMru0dlwB8zu1OoErOazZoc8ISSJn" -O LV-MHP-v2.zip
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
