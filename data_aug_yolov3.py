#coding=utf-8

"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

Ellis Brown, Max deGroot
"""

import torch
import cv2
import numpy as np
import random
import math
from lib.utils.box_utils import matrix_iou




def _crop(image, boxes, labels):
    height, width, _ = image.shape

    if len(boxes)== 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.5, 1.)
            # min_ratio = max(0.5, scale*scale)
            # max_ratio = min(2, 1. / scale / scale)
            # ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * width)
            h = int(scale * height)


            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])

            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                     .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t,labels_t


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp
    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
    if random.randrange(2):
        _convert(image[:, :, 2], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes, fill, p):
    if random.random() > p:
        return image, boxes

    height, width, depth = image.shape
    for _ in range(1):
        # scale = random.uniform(1,4)
        #
        # min_ratio = max(0.5, 1./scale/scale)
        # max_ratio = min(2, scale*scale)
        # ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ratio = random.uniform(1, 2)
        # ws = scale*ratio
        # hs = scale/ratio
        ws = ratio
        hs = ratio
        if ws < 1 or hs < 1:
            continue
        tmp_max_val = max(width, height)
        w = int(ws * tmp_max_val)
        h = int(hs * tmp_max_val)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)


        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes

def _rotate(image, boxes):
    height, width, _ = image.shape
    degree = random.randint(0,8)
    #0
    if degree <6:
        image = image
        boxes_t = boxes.copy()
    #90
    if degree ==6:
       #image = cv2.flip(image,-1)
       image = cv2.transpose(image)
       image = cv2.flip(image,0)
       boxes = boxes.copy()
       boxes_t = boxes.copy()
       boxes_t[:, 0] = boxes[:, 1]
       boxes_t[:, 1] = width - boxes[:, 2]
       boxes_t[:, 2] = boxes[:, 3]
       boxes_t[:, 3] = width - boxes[:, 0]
    #180
    if degree ==7:
       image = cv2.flip(image,-1)
       boxes = boxes.copy()
       boxes_t = boxes.copy()
       boxes_t[:, 0] = width - boxes[:, 2]
       boxes_t[:, 1] = height - boxes[:, 3]
       boxes_t[:, 2] = width - boxes[:, 0]
       boxes_t[:, 3] = height - boxes[:, 1]
    #270
    if degree ==8:
       image = cv2.transpose(image)
       image = cv2.flip(image,1)
       boxes = boxes.copy()
       boxes_t = boxes.copy()
       boxes_t[:, 0] = height - boxes[:, 3]
       boxes_t[:, 1] = boxes[:, 0]
       boxes_t[:, 2] = height - boxes[:, 1]
       boxes_t[:, 3] = boxes[:, 2]
    return image, boxes_t


def _elastic(image, p, alpha=None, sigma=None, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
     From:
     https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
    """
    if random.random() > p:
        return image
    if alpha == None:
        alpha = image.shape[0] * random.uniform(0.5,2)
    if sigma == None:
        sigma = int(image.shape[0] * random.uniform(0.5,1))
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape[:2]

    dx, dy = [cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1) * alpha, (sigma|1, sigma|1), 0) for _ in range(2)]
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    x, y = np.clip(x+dx, 0, shape[1]-1).astype(np.float32), np.clip(y+dy, 0, shape[0]-1).astype(np.float32)
    return cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR, borderValue= 0, borderMode=cv2.BORDER_REFLECT)


# 随机旋转某个角度的数据增强方式
def _rand_rotate(img,boxes,angle,scale=1.):
    w,h=img.shape[0],img.shape[1]
    rangle=np.deg2rad(angle)

    nw=(abs(np.sin(rangle)*h)+abs(np.cos(rangle)*w))*scale
    nh=(abs(np.cos(rangle)*h)+abs(np.sin(rangle)*w))*scale

    rot_mat=cv2.getRotationMatrix2D((nw*0.5,nh*0.5),angle,scale)
    rot_move=np.dot(rot_mat,np.array([(nw-w)*0.5,(nh-h)*0.5,0]))

    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]

    img=cv2.warpAffine(img,rot_mat,(int(math.ceil(nw)),int(math.ceil(nh))),flags=cv2.INTER_LANCZOS4)

    coord_bboxes = []
    for bbox in boxes:
        xmin, ymin, xmax, ymax = bbox
        point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
        # concat np.array
        concat = np.vstack((point1, point2, point3, point4))
        # change type
        concat = concat.astype(np.int32)
        rx, ry, rw, rh = cv2.boundingRect(concat)

        xmin = rx
        ymin = ry
        xmax = rx + rw
        ymax = ry + rh
        bbox = [xmin, ymin, xmax, ymax]
        coord_bboxes.append(bbox)

    coord_bboxes=np.asarray(coord_bboxes,dtype=np.float32)
    return img,coord_bboxes


def preproc_for_test(image, insize, mean, is_random = True):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    if is_random is not True:
        interp_method = interp_methods[2]
    image = cv2.resize(image, (insize[0], insize[1]),interpolation=interp_method)
    image = image.astype(np.float32)
    image -= mean
    return image.transpose(2, 0, 1)

def draw_bbox(image, bbxs, color=(0, 255, 0)):
    img = image.copy()
    bbxs = np.array(bbxs).astype(np.int32)
    for bbx in bbxs:
        cv2.rectangle(img, (bbx[0], bbx[1]), (bbx[2], bbx[3]), color, 5)
    return img

class preproc(object):

    def __init__(self, resize, rgb_means, p, writer=None):
        self.means = rgb_means
        self.resize = resize
        self.p = p
        self.writer = writer # writer used for tensorboard visualization
        self.epoch = 0

    def __call__(self, image, targets=None):
        # some bugs
        if self.p == -2: # abs_test
            targets = np.zeros((1,5))
            targets[0] = image.shape[0]
            targets[0] = image.shape[1]
            image = preproc_for_test(image, self.resize, self.means, is_random = False)
            return torch.from_numpy(image), targets

        boxes = targets[:,:-1].copy()
        labels = targets[:,-1].copy()
        if len(boxes) == 0:
            targets = np.zeros((1,5))
            image = preproc_for_test(image, self.resize, self.means) # some ground truth in coco do not have bounding box! weird!
            return torch.from_numpy(image), targets
        if self.p == -1: # eval
            height, width, _ = image.shape
            boxes[:, 0::2] /= width
            boxes[:, 1::2] /= height
            labels = np.expand_dims(labels,1)
            targets = np.hstack((boxes,labels))
            image = preproc_for_test(image, self.resize, self.means)
            return torch.from_numpy(image), targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:,:-1]
        labels_o = targets_o[:,-1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o,1)
        targets_o = np.hstack((boxes_o,labels_o))

        if self.writer is not None:
            image_show = draw_bbox(image, boxes)
            self.writer.add_image('preprocess/input_image', image_show, self.epoch)

        image_t, boxes, labels = _crop(image, boxes, labels)
        # print("crop", image_t.shape)
        if self.writer is not None:
            image_show = draw_bbox(image_t, boxes)
            self.writer.add_image('preprocess/crop_image', image_show, self.epoch)

        image_t = _distort(image_t)
        # print("_distort", image_t.shape)
        if self.writer is not None:
            image_show = draw_bbox(image_t, boxes)
            self.writer.add_image('preprocess/distort_image', image_show, self.epoch)

        # image_t = _elastic(image_t, self.p)
        # if self.writer is not None:
        #     image_show = draw_bbox(image_t, boxes)
        #     self.writer.add_image('preprocess/elastic_image', image_show, self.epoch)

        # image_t, boxes = _expand(image_t, boxes, self.means, self.p)
        # # print("_expand", image_t.shape)
        # if self.writer is not None:
        #     image_show = draw_bbox(image_t, boxes)
        #     self.writer.add_image('preprocess/expand_image', image_show, self.epoch)

        image_t, boxes = _mirror(image_t, boxes)
        # print("_mirror", image_t.shape)
        if self.writer is not None:
            image_show = draw_bbox(image_t, boxes)
            self.writer.add_image('preprocess/mirror_image', image_show, self.epoch)

        image_t, boxes = _rotate(image_t, boxes)
        # print("_rotate", image_t.shape)
        if self.writer is not None:
            image_show = draw_bbox(image_t, boxes)
            self.writer.add_image('preprocess/rotate_image', image_show, self.epoch)

        angle=np.random.randint(0,360,1)
        image_t,boxes=_rand_rotate(image_t,boxes,angle=angle)
        if self.writer is not None:
            img_show=draw_bbox(image_t,boxes)
            self.writer.add_image("preprocess/rand_rotate_imag",img_show,self.epoch)



        # only write the preprocess step for the first image
        # if self.writer is not None:
        #     # print('image adding')
        #     self.release_writer()

        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.resize, self.means)
        # print("preproc_for_test", image_t.shape)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b= np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()


        if len(boxes_t)==0:
            image = preproc_for_test(image_o, self.resize, self.means)
            return torch.from_numpy(image),targets_o

        labels_t = np.expand_dims(labels_t,1)
        targets_t = np.hstack((boxes_t,labels_t))

        return torch.from_numpy(image_t), targets_t
    
    def add_writer(self, writer, epoch=None):
        self.writer = writer
        self.epoch = epoch if epoch is not None else self.epoch + 1
    
    def release_writer(self):
        self.writer = None
