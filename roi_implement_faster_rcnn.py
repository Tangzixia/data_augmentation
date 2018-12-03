### 复制：https://blog.csdn.net/wuguangbin1230/article/details/80019818 
'''
通过　tf.image.sample_distorted_bounding_box函数可以进行裁剪，根据对应的位置对图片进行裁剪～
tf.image.sample_distortd_bounding_box解析：https://blog.csdn.net/tz_zs/article/details/77920116
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
       
       
  '''
  ('bbox: ', array([[[0.19375   , 0.1015625 , 0.90208334, 0.775     ]]], dtype=float32))
('bbox_begin: ', array([93, 65,  0], dtype=int32))
('bbox_size: ', array([340, 431,  -1], dtype=int32))
('bbox: ', array([[[0.00208333, 0.103125  , 0.79791665, 0.6       ]]], dtype=float32))
('bbox_begin: ', array([ 1, 66,  0], dtype=int32))
('bbox_size: ', array([382, 318,  -1], dtype=int32))
('bbox: ', array([[[0.16041666, 0.0484375 , 0.90208334, 0.6828125 ]]], dtype=float32))
('bbox_begin: ', array([77, 31,  0], dtype=int32))
('bbox_size: ', array([356, 406,  -1], dtype=int32))
('bbox: ', array([[[0.18125  , 0.3828125, 0.8541667, 0.990625 ]]], dtype=float32))
('bbox_begin: ', array([ 87, 245,   0], dtype=int32))
('bbox_size: ', array([323, 389,  -1], dtype=int32))
('bbox: ', array([[[0.15      , 0.121875  , 0.72083336, 0.64375   ]]], dtype=float32))
('bbox_begin: ', array([72, 78,  0], dtype=int32))
('bbox_size: ', array([274, 334,  -1], dtype=int32))
('bbox: ', array([[[0.13958333, 0.33125   , 0.95      , 0.90625   ]]], dtype=float32))
('bbox_begin: ', array([ 67, 212,   0], dtype=int32))
('bbox_size: ', array([389, 368,  -1], dtype=int32))
  '''
