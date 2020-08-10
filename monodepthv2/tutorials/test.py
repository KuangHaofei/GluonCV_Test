import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib import pyplot as plt


if __name__ == '__main__':
    input_img = 'input_img.png'
    input_stereo_img = 'input_stereo_img.png'
    input_gt = 'input_gt.png'

    input_img = pil.open(input_img).convert('RGB')
    input_stereo_img = pil.open(input_stereo_img).convert('RGB')
    input_gt = pil.open(input_gt)

    fig = plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.75)
    fig.add_subplot(3, 1, 1)
    plt.title("left image")
    plt.imshow(input_img)

    fig.add_subplot(3, 1, 2)
    plt.title("right image")
    plt.imshow(input_stereo_img)

    fig.add_subplot(3, 1, 3)
    plt.title("ground truth of left input (the reprojection of LiDAR data)")
    plt.imshow(input_gt)

    plt.show()
