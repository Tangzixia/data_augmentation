#coding=utf-8

import cv2
import numpy as np
import math
import random

img_path="/home/hp/Pictures/train_loss.png"
rand_angle=180

min_scale=0.5
max_scale=2.5
### 对图像做了数据增强，那么标签怎么判断呢？

### 关于旋转的数据增强，注意旋转后对应的标签也会发生变化，但仍旧保持正的～
### 参考：https://blog.csdn.net/uncle_ll/article/details/83930861，
### 更多数据增强方式参考：
###     https://github.com/maozezhong/CV_ToolBox/blob/master/DataAugForObjectDetection/DataAugmentForObejctDetection.py
### 其中cv2.boundingRect(concat)用于找出最小的矩形，用于将对应的点包含在内
def rotate_img(img,rand_angle):
    orig_h,orig_w=img.shape[:2]
    # rand_angle=np.random.randint(0,rand_angle)
    # print(rand_angle)

    rangle = np.deg2rad(rand_angle)
    nw = (abs(np.sin(rangle) * orig_h) + abs(np.cos(rangle) * orig_w)) * 1.0
    nh = (abs(np.cos(rangle) * orig_h) + abs(np.sin(rangle) * orig_w)) * 1.0

    rot_mat = cv2.getRotationMatrix2D((orig_h / 2, orig_w / 2), rand_angle, 1)
    rot_move = np.dot(rot_mat, np.array([(nw - orig_w) * 0.5, (nh - orig_h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]




    img_rotate = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
            flags=cv2.INTER_LANCZOS4)
    return img_rotate

def rotate_image_std(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    angle=np.random.randint(0,angle)
    # convet angle into rad
    rangle = np.deg2rad(angle)  # angle in radians
    # calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # map
    return cv2.warpAffine(
        src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
        flags=cv2.INTER_LANCZOS4)

def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
    return image

def scale_image(img,scale):
    orig_h, orig_w = img.shape[:2]
    return cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))


if __name__=="__main__":
    img=cv2.imread(img_path)
    orig_h,orig_w=img.shape[:2]

    img_rotate=rotate_image_std(img,rand_angle)
    # img_scale=scale_image(img, min_scale + np.random.rand() * (max_scale - min_scale))
    # print(img_rotate.shape[:2])

    cv2.imshow("hello.png",img_rotate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
