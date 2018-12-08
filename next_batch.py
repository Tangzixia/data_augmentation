#coding=utf-8

import numpy as np
import numpy
import cv2
import os

def batch_iter(sourceData, batch_size, num_epochs, shuffle=False):
    data = np.array(sourceData)  # 将sourceData转换为array存储
    data_size = len(sourceData)
    num_batches_per_epoch = int(len(sourceData) / batch_size) + 1
    for num_epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = sourceData[shuffle_indices]
        else:
            shuffled_data = sourceData

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]

def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width, output_height,
                               output_width):
    images = os.listdir(images_path)
    segmentations = os.listdir(segs_path)
    images.sort()
    segmentations.sort()
    for i in range(len(images)):
        images[i] = images_path + images[i]
        segmentations[i] = segs_path + segmentations[i]

    assert len(images) == len(segmentations)
    for im, seg in zip(images, segmentations):

        assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0])

    zipped = itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            X.append(getImageArr(im, input_width, input_height))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)
        
if __name__=="__main__":
    tst_dirs="/home/jobs/code/unet-master/data/membrane/test"
    files=os.listdir(tst_dirs)
    data=[]
    for file in files:
        file_cur=os.path.join(tst_dirs,file)
        img=cv2.imread(file_cur)
        img=np.asarray(img,dtype=np.float32)
        img=cv2.resize(img,(256,256))
        data.append(img)
    print("--------0---------")
    data_gener = batch_iter(data, batch_size=5, num_epochs=20, shuffle=False)
    for iter in range(200):
        batch_data=next(data_gener)
        print(batch_data)
