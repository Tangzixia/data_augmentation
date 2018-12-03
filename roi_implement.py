### 复制：https://blog.csdn.net/wuguangbin1230/article/details/80019818 
'''
通过tf.image.sample_distorted_bounding_box函数可以进行裁剪，根据对应的位置对图片进行裁剪～
'''

import tensorflow as tf
import matplotlib.pyplot as plt
 
image_raw_data = tf.gfile.FastGFile("/data/Data/cornell_grasping_dataset/process_cornell_data/all_png/pcd0100r.png",'r').read()
with tf.Session() as Sess:
    ima_data = tf.image.decode_png(image_raw_data)
    image_float = tf.image.convert_image_dtype(ima_data,dtype=tf.float32)
    bbox = [251.0/480.,254.0/640,369.0/480,310.0/640]
    boxes = tf.constant([[bbox]])
 
    original_image = tf.expand_dims(image_float, 0)
    image_with_box = tf.image.draw_bounding_boxes(original_image,boxes=boxes)
    plt.figure(1)
    plt.imshow(image_with_box.eval().reshape([480, 640, 3]))
    plt.show()
 
    for i in range(6):
        bbox_begin, bbox_size, bbox_2 = tf.image.sample_distorted_bounding_box(tf.shape(image_float), 
                                                                                bounding_boxes=boxes,
                                                                               min_object_covered=0.3,
                                                                               aspect_ratio_range=[0.75, 1.33])
        distort_image = tf.slice(image_float, bbox_begin, bbox_size)
        distort_image = tf.expand_dims(distort_image, dim=0)
        image_with_box = tf.image.draw_bounding_boxes(original_image, boxes=bbox_2)
