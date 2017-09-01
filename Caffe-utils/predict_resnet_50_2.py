#!usr/bin/env python
# -*- coding:utf-8 -*-

# Extract Features of Certain Layers by Using Pretrained ResNet-50

# Files needed:
#   Input Image, e.g. 1_1_BACK_LEFT.png
#   Pretrained Caffe Model, e.g. ResNet-50-model.caffemodel
#   prototxt file, e.g. ResNet-50-deploy.prototxt

# Requirements:
#   pycaffe '1.0.0'
#   Python 2.7.12 (default, Nov 19 2016, 06:48:10). (Because Caffe only supports Python 2)
#   [GCC 5.4.0 20160609] on linux2
#   numpy '1.13.1'
#   h5py '2.7.0'
#   PIL '4.2.1'
#   matplotlib '2.0.2'


# Reference: [1] http://blog.csdn.net/tina_ttl/article/details/51033660
#            [2] http://lijiancheng0614.github.io/2015/08/21/2015_08_21_CAFFE_Features/
#            [3] https://github.com/BVLC/caffe/blob/master/python/caffe/pycaffe.py
#            [4] http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
#            [5] https://github.com/KaimingHe/deep-residual-networks
#	         [6] http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006
#            [7] http://www.jianshu.com/p/9644f7ec0a03


import caffe
import os
import numpy as np
import h5py

import matplotlib.pyplot as plt
from PIL import Image

def location():
    
    files = os.listdir(path)
    arr_location = np.zeros((100, 100))
    x_max, y_max = 0, 0

    # Traverse the files
    for file in files:
        if not os.path.isdir(file) and os.path.splitext(file)[1] == ".png":
            x,y = int(file.split('_')[0]), int(file.split('_')[1])

            if x>x_max:
                x_max = x
            if y>y_max:
                y_max = y
            arr_location[x,y] = 1

    print("x_max: ", x_max, "y_max: ", y_max)

    # Save as HDF5 file
    dset = f.create_dataset("location", data = arr_location)

# Mapping between the feature and the corresponding picture
def location_order():
    arr_location = np.full((100, 100), -1)
    x_max, y_max = 0, 0
    counter = -4

    for x in range(100):
        for y in range(100):
            if os.path.isfile(path+str(x)+'_'+str(y)+'_FRONT_LEFT.png'):
                counter = counter + 4

                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                arr_location[x,y] = counter

    print("x_max: ", x_max, "y_max: ", y_max)

    # Save as HDF5 file
    dset = f.create_dataset("location", data = arr_location)

def vis_square(data):

    # 输入的数据为一个ndarray，尺寸可以为(n, height, width)或者是 (n, height, width, 3)
    # 前者即为n个灰度图像的数据，后者为n个rgb图像的数据
    # 在一个sqrt(n) by sqrt(n)的格子中，显示每一幅图像

    # 对输入的图像进行normlization
    data = (data - data.min()) / (data.max() - data.min())

    # 强制性地使输入的图像个数为平方数，不足平方数时，手动添加几幅
    n = int(np.ceil(np.sqrt(data.shape[0])))
    # 每幅小图像之间加入小空隙
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
                           + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)   

    # 将所有输入的data图像平复在一个ndarray-data中（tile the filters into an image）
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    # data的一个小例子,e.g., (3,120,120)
    # 即，这里的data是一个2d 或者 3d 的ndarray
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    # 显示data所对应的图像
    plt.imshow(data)
    plt.axis('off')
    plt.show()

def feature():

    net=caffe.Net(
        "ResNet-50-deploy.prototxt",
        "ResNet-50-model.caffemodel",
        caffe.TEST
    )

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

    arr_feature = np.zeros([0, 0])
    print(arr_feature.shape)
    counter = 0

    for x in range(100):
        for y in range(100):
            if os.path.isfile(path+str(x)+'_'+str(y)+'_FRONT_LEFT.png'):
                # FRONT_LEFT
                image=caffe.io.load_image(path+str(x)+'_'+str(y)+'_FRONT_LEFT.png')
                net.blobs['data'].data[...] = transformer.preprocess('data', image)
                
                result = net.forward()

                feat = net.blobs['pool5'].data[0]
                feat = feat.reshape([2048, 1])
                print(feat)
                # Append values to the end of an array
                arr_feature = np.append(arr_feature, feat)

                counter = counter+1
                print(path+str(x)+'_'+str(y)+'_FRONT_LEFT.png')

                # BACK_LEFT
                image=caffe.io.load_image(path+str(x)+'_'+str(y)+'_BACK_LEFT.png')
                net.blobs['data'].data[...] = transformer.preprocess('data', image)
                
                result = net.forward()

                feat = net.blobs['pool5'].data[0]
                feat = feat.reshape([2048, 1])
                print(feat)
                # Append values to the end of an array
                arr_feature = np.append(arr_feature, feat)

                counter = counter+1
                print(path+str(x)+'_'+str(y)+'_BACK_LEFT.png')

                # LEFT_LEFT
                image=caffe.io.load_image(path+str(x)+'_'+str(y)+'_LEFT_LEFT.png')
                net.blobs['data'].data[...] = transformer.preprocess('data', image)
                
                result = net.forward()

                feat = net.blobs['pool5'].data[0]
                feat = feat.reshape([2048, 1])
                print(feat)
                # Append values to the end of an array
                arr_feature = np.append(arr_feature, feat)

                counter = counter+1
                print(path+str(x)+'_'+str(y)+'_LEFT_LEFT.png')

                # RIGHT_LEFT
                image=caffe.io.load_image(path+str(x)+'_'+str(y)+'_RIGHT_LEFT.png')
                net.blobs['data'].data[...] = transformer.preprocess('data', image)
                
                result = net.forward()

                feat = net.blobs['pool5'].data[0]
                feat = feat.reshape([2048, 1])
                print(feat)
                # Append values to the end of an array
                arr_feature = np.append(arr_feature, feat)

                counter = counter+1
                print(path+str(x)+'_'+str(y)+'_RIGHT_LEFT.png')

    arr_feature = arr_feature.reshape([2048, counter])
    dset = f.create_dataset("resnet_feature", data=arr_feature)

if __name__=='__main__':

    path = ("/home/jack/Applications/icra2017/data/apartment_2/1/")  # Target Directory
    f = h5py.File('apartment_2.h5', 'a')
    # delete dataset 'location'
    # del f['location']
    feature()
    location_order()
