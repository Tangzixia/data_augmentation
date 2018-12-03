### 复制：https://blog.csdn.net/tiandd12/article/details/80105430 

#encoding:utf-8
'''
tf 参考链接 ：https://tensorflow.google.cn/api_guides/python/image
增加数据量，减轻过拟合，增强模型的泛化能力
在预测时也可以使用
'''
import numpy as np
import os
import math
import tensorflow as tf
from skimage import io
import random
import matplotlib.pyplot as plt
 
def read_image(image_path):
	image_raw_data = tf.gfile.FastGFile(image_path,'rb').read()
	image_data = tf.image.decode_png(image_raw_data)
	return image_data
 
'''
	#图像大小的调整,放大缩小
	不同尺寸
	tf.image.resize_images(img,size,size,method), 0,默认 双线性插值；1，最近邻算法；
                                              2， 双3次插值法；3，面积插值法
'''
def resize_image(image_data):
	
	res = []
	image_biliner = tf.image.resize_images(image_data,[256,256],method=0)	
 
	image_nn = tf.image.resize_images(image_data,[256,256],method=1)
	image_bicubic = tf.image.resize_images(image_data,[256,256],method=2)
	image_area = tf.image.resize_images(image_data,[256,256],method=3)
 
	res.append(tf.to_int32(image_biliner))
	res.append(tf.to_int32(image_nn))
	res.append(tf.to_int32(image_bicubic))
	res.append(tf.to_int32(image_area))
	
	return res
 
'''
#裁剪
识别不同位置的物体
'''
def crop_image(image_data):
	res = []
	#在中间位置进行裁剪或者周围填充0
	image_crop = tf.image.resize_image_with_crop_or_pad(image_data,256,256)	
	image_pad = tf.image.resize_image_with_crop_or_pad(image_data,512,512)
	
	#按照比列 裁剪图像的中心区域
	image_center_crop = tf.image.central_crop(image_data,0.5)
 
	#随机裁剪（常用方法）
	image_random_crop0 = tf.random_crop(image_data,[300,300,3])
	image_random_crop1 = tf.random_crop(image_data,[300,300,3])
 
	res.append(tf.to_int32(image_crop))
	res.append(tf.to_int32(image_pad))
	res.append(tf.to_int32(image_center_crop))
	res.append(tf.to_int32(image_random_crop0))	
	res.append(tf.to_int32(image_random_crop1))
 
	return res
 
'''
	#旋转（镜像）
	图像旋转不会影响识别的结果，可以在多个角度进行旋转，使模型可以识别不同角度的物体
	当旋转或平移的角度较小时，可以通过maxpooling来保证旋转和平移的不变性。
'''
def flip_image(image_data):
 
	#镜像
	res = []
	#上下翻转
	image_up_down_flip = tf.image.flip_up_down(image_data)
 
	#左右翻转
	image_left_right_filp = tf.image.flip_left_right(image_data)
 
	#对角线旋转
	image_transpose = tf.image.transpose_image(image_data)
 
	#旋转90度 
	image_rot1 = tf.image.rot90(image_data,1)
	image_rot2 = tf.image.rot90(image_data,2)
	image_rot3 = tf.image.rot90(image_data,3)
 
	res.append(tf.to_int32(image_up_down_flip))
	res.append(tf.to_int32(image_left_right_filp))
	res.append(tf.to_int32(image_transpose))
	res.append(tf.to_int32(image_rot1))
	res.append(tf.to_int32(image_rot2))
	res.append(tf.to_int32(image_rot3))
 
	return res
 
#图像色彩调整
'''
	根据原始数据模拟出更多的不同场景下的图像
	brightness(亮度),适应不同光照下的物体
	constrast（对比度）, hue（色彩）, saturation（饱和度） 
	可自定义和随机
'''
def color_image(image_data):
	res = []
 
	image_random_brightness = tf.image.random_brightness(image_data,0.5)
	image_random_constrast = tf.image.random_contrast(image_data,0,1)
	image_random_hue = tf.image.random_hue(image_data,0.5)
	image_random_saturation = tf.image.random_saturation(image_data,0,1)
 
	#颜色空间变换
	images_data = tf.to_float(image_data)
	image_hsv_rgb = tf.image.rgb_to_hsv(images_data)
	# image_gray_rgb = tf.image.rgb_to_grayscale(image_data)
	# image_gray_rgb = tf.expand_dims(image_data[2],1)
 
	res.append(tf.to_int32(image_random_brightness))
	res.append(tf.to_int32(image_random_constrast))
	res.append(tf.to_int32(image_random_hue))
	res.append(tf.to_int32(image_random_saturation))
	res.append(tf.to_int32(image_hsv_rgb))
	return res
 
#添加噪声
def PCA_Jittering(img):
 
	img_size = img.size/3
	print(img.size,img_size)
	img1= img.reshape(int(img_size),3)
	img1 = np.transpose(img1)
	img_cov = np.cov([img1[0], img1[1], img1[2]])  
	#计算矩阵特征向量
	lamda, p = np.linalg.eig(img_cov)
 
	p = np.transpose(p)  
	#生成正态分布的随机数
	alpha1 = random.normalvariate(0,0.2)  
	alpha2 = random.normalvariate(0,0.2)  
	alpha3 = random.normalvariate(0,0.2)  
 
	v = np.transpose((alpha1*lamda[0], alpha2*lamda[1], alpha3*lamda[2])) #加入扰动  
	add_num = np.dot(p,v)  
 
	img2 = np.array([img[:,:,0]+add_num[0], img[:,:,1]+add_num[1], img[:,:,2]+add_num[2]])  
 
	img2 = np.swapaxes(img2,0,2)  
	img2 = np.swapaxes(img2,0,1)  
 
	return img2
 
def main(_):
	image_path = 'dog.png'
	image_data = read_image(image_path)
	img = tf.image.per_image_standardization(image_data)
	resize = resize_image(image_data)
	crop = crop_image(image_data)
	flip = flip_image(image_data)
	color = color_image(image_data)
	init = tf.global_variables_initializer()
 
	with tf.Session() as sess:
		sess.run(init)
		img, resize_res, crop_res, flip_res, color_res = sess.run([img,
			resize,crop,flip,color])
		
		res = []
		res.append(resize_res)
		res.append(crop_res)
		res.append(flip_res)
		res.append(color_res)
 
		for cat in res:
			fig = plt.figure()
			num = 1
			for i in cat:	
				x = math.ceil(len(cat)/2) #向上取整				
				fig.add_subplot(2,x,num)	
				plt.imshow(i)
				num = num+1
			plt.show()
		img = PCA_Jittering(img)
		plt.imshow(img)
		plt.show()
if __name__ == '__main__':
	
	tf.app.run()


