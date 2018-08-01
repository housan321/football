# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 20:44:13 2018

@author: Administrator
"""
from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from sklearn.externals import joblib
import os

import numpy as np

import dl_data_process as dp


num_classes=3

load_dir = os.path.join(os.getcwd(), 'saved_models')
model_mlp = 'mlp_model.h5'
model_cnn = 'cnn_model.h5'
model_svm = 'svm_model731.m'


# load model and weights
if not os.path.isdir(load_dir):
    os.makedirs(load_dir)
model_path_mlp = os.path.join(load_dir, model_mlp)
model_path_cnn = os.path.join(load_dir, model_cnn)
model_path_svm = os.path.join(load_dir, model_svm)



files = ['odds2018(no_nor1).dat','odds2018(no_nor1).dat']
# The data, split between train and test sets.
(x_train, y_train), (x_test, y_test) = dp.load_data(files)


## svm
(x_train_svm, y_train_svm), (x_test_svm, y_test_svm) = dp.load_data_svm(files)


## mlp
#x_train_mlp = x_train.reshape(x_train.shape[0], -1)
x_test_mlp = x_test.reshape(x_test.shape[0], -1)
#x_train_mlp = x_train_mlp.astype('float32')
x_test_mlp = x_test_mlp.astype('float32')

## cnn
#x_train_cnn = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test_cnn = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
#x_train_cnn = x_train_cnn.astype('float32')
x_test_cnn = x_test_cnn.astype('float32')




model_mlp = load_model(model_path_mlp)
model_cnn = load_model(model_path_cnn)
model_svm = joblib.load(model_path_svm)

 
   
y_pre_mlp = model_mlp.predict(x_test_mlp)
y_pre_cnn = model_cnn.predict(x_test_cnn)
y_pre_svm = model_svm.predict_proba(x_test_svm)



y_cls_mlp = model_mlp.predict_classes(x_test_mlp)
y_cls_cnn = model_cnn.predict_classes(x_test_cnn)
y_cls_svm = model_svm.predict(x_test_svm)



rate_mlp = np.mean(y_cls_mlp.ravel() == y_test.ravel()) * 100
rate_cnn = np.mean(y_cls_cnn.ravel() == y_test.ravel()) * 100
rate_svm = np.mean(y_cls_svm.ravel() == y_test_svm.ravel()) * 100


'''#不用的
mlp_mask = (y_cls_mlp==y_test)
right_mlp = y_cls_mlp[mlp_mask]

cnn_mask = (y_cls_cnn==y_train)
right_cnn = y_cls_cnn[cnn_mask]
'''


y_pre2 = (y_pre_mlp + y_pre_cnn)/2
y_cls2 = y_pre2.argmax(axis=1)
mask2 = (y_cls2==y_test)
right2 = y_cls2[mask2]



'''
y_pre3 = (y_pre_mlp + y_pre_cnn + y_pre_svm)/3
y_cls3 = y_pre3.argmax(axis=1)
mask3 = (y_cls3==y_train)
right3 = y_cls3[mask3]
'''



print('mlp深度学习的准确率为：', rate_mlp)
print('cnn深度学习的准确率为：', rate_cnn)
print('svm深度学习的准确率为：', rate_svm)
print('mlp & cnn合并深度学习的准确率为：', len(right2)/len(y_test))
#print('mlp & cnn & svm 合并深度学习的准确率为：', len(right3)/len(y_train))


k=1


