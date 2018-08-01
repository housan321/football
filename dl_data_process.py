# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 18:10:23 2018

@author: Administrator


深度学习赔率数据

"""

import os,sys,re
import arrow,bs4
import pandas as pd

import requests
from bs4 import BeautifulSoup 


import zsys
import ztools as zt
import ztools_str as zstr
import ztools_web as zweb
import ztools_data as zdat
import zpd_talib as zta
#
import tfb_sys as tfsys
import tfb_tools as tft
import tfb_strategy as tfsty


import numpy as np

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

from sklearn.cross_validation import train_test_split

TIME_STEPS = 31     # same as the height of the image
INPUT_SIZE = 6     # same as the width of the image
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 3
CELL_SIZE = 50
LR = 0.001





#1
rs0='/tfbDat/'
fgid,fxdat=rs0+'gid2018.dat', rs0+'xdat2018.dat'




#############################################################################################
########################################################################################

def normalize_odds_oz(odds_oz):
    nor_odds = pd.DataFrame(columns=tfsys.usedSgn_oz)
    odds_oz = odds_oz.astype(float)
    odds_oz = odds_oz-1
    for i, row in odds_oz.iterrows():
        row = row / row.max()
        nor_odds = nor_odds.append(row)
    
    return nor_odds
        
        
def normalize_odds_az(odds_az):
    nor_odds = pd.DataFrame(columns=tfsys.usedSgn_az)
    odds_az = odds_az.astype(float)
    for i, row in odds_az.iterrows():
        row = row / row.max()
        nor_odds = nor_odds.append(row)
    
    return nor_odds         
   
      
def normalize_odds_tz(odds_tz): 
    odds_tz = odds_tz.astype(float)
    odds_tz = odds_tz/100
    odds_tz[odds_tz>1] = 1
    
    return odds_tz
    

# 过滤让球过大的比赛，让球大于2球的比赛不作样本
def  is_filter_sample(odds_az):
    rq = odds_az[['mrangqui0','grangqui0','mrangqui9','grangqui9']]
    rq = rq.astype(float)

    if rq.iat[0,0]>=1.75 or rq.iat[0,1]>=1.75 or rq.iat[0,2]>=1.75 or rq.iat[0,3]>=1.75 : 
        return True  
    else: 
        return False



'''
提取每场所比赛的数据（亚盘赔率、欧盘赔率、投注情况等），并保存在一个文件
file_name:保存文件名
num：比赛数量
'''
def save_odds_to_file(file_name, num):
    #2---init.tfb
    xtfb = tft.fb_init(rs0, fgid)  
    df =  tfsys.gids 
    p_data = pd.DataFrame()
#    p_data_oz = pd.DataFrame()
#    p_data_az = pd.DataFrame()
    
    for i, row in df.iterrows():
        if ((i+1) % 2000 == 0):
            print((i+1)/len(df) * 100, "%")
            print('now:',zt.tim_now_str())
        if i>=num[0] and i<num[1]:    
            gid = row['gid']
            fxdat_oz = tfsys.rxdat + gid + '_oz.dat'
            fxdat_az = tfsys.rxdat + gid + '_az.dat'
            fxdat_tz = tfsys.rxdat + gid + '_tz.dat'
            
            if os.path.exists(fxdat_oz) and os.path.exists(fxdat_az) and os.path.exists(fxdat_tz):
                odds_oz = pd.read_csv(fxdat_oz, index_col = False, dtype = str, encoding = 'gb18030')  
                odds_az = pd.read_csv(fxdat_az, index_col = False, dtype = str, encoding = 'gb18030')  
                odds_tz = pd.read_csv(fxdat_tz, index_col = False, dtype = str, encoding = 'gb18030')  
                
                if len(odds_oz) >= tfsys.cidrows \
                        and odds_oz.loc[0, 'kwin'] != '-1' \
                        and len(odds_az) >= tfsys.cidrows \
                        and odds_az.loc[0, 'kwin'] != '-1' \
                        and len(odds_tz) == 1:    #如果数据有CID_ROWS行并且有比赛结果才处理数据

                    
                    odds_oz = odds_oz[odds_oz['cid'] == '3']
                    odds_az = odds_az[odds_az['cid'] == '3']
                    
                    if odds_oz.empty or odds_az.empty:
                        continue
                    
                  
                    label = odds_oz['kwin']
                    
                    odds_oz = odds_oz[tfsys.usedSgn_oz]
                    odds_az = odds_az[tfsys.usedSgn_az]
                    odds_tz = odds_tz[tfsys.usedSng_tz]
                    odds_oz = odds_oz.reset_index(drop=True)
                    odds_az = odds_az.reset_index(drop=True)
                    odds_tz = odds_tz.reset_index(drop=True)
                    label = label.reset_index(drop=True)
                    
#                    odds_oz = odds_oz.astype(float)
#                    odds_az = odds_az.astype(float)
                                        
                    
                                       
                    flag = is_filter_sample(odds_az)
                    if flag: 
                        continue
                                      

                    merge_data = pd.concat([odds_az, odds_oz, odds_tz, label], axis=1)                 
                    p_data = p_data.append(merge_data, ignore_index=True)

    p_data.to_csv(file_name, index=False, encoding='gb18030')



#####################################################################################
##训练读入数据
'''
读入赔率文件
'''
def load_odds_file(files):
    kwin = []
    data = pd.DataFrame()
    for file_n in files:
        dt = pd.read_csv(file_n)
        data = data.append(dt)

    data.astype('float32')    
    for row in range(0, len(data), tfsys.cidrows):
        odds = data[row:row+tfsys.cidrows]
        kwin.append((odds.iloc[0][29]).astype(int))   #提取每场比赛的kwin
    data.pop('kwin')

    kwin = np.array(kwin)
    return kwin, data


def load_odds_file_svm(files):
    kwin = []
    data = pd.DataFrame()
    for file_n in files:
        dt = pd.read_csv(file_n)
        data = data.append(dt)
    data.astype('float32')

    kwin = data['kwin']
    kwin = kwin.astype(int)  
    data.pop('kwin')
      
#    data1 = data[1:len(data):tfsys.cidrows]
#    kwin1 = kwin[1:len(data):tfsys.cidrows]
    
    return kwin, data

    

        

'''
归一化数据
'''
def reshape_data(data):
    nor_data = []
#    row = int(len(data)/tfsys.cidrows)
    
    for row in range(0, len(data), tfsys.cidrows):
        odds = data[row:row+tfsys.cidrows]
        arr_odds = np.array(odds)  

        nor_data.append(arr_odds)
    nor_data = np.array(nor_data)     #转化为np.array数组
    return nor_data



def load_data(files):
    target, data = load_odds_file(files)

    data = reshape_data(data)    


    # split into train and test sets
    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    x_train, x_test = data[0:train_size,:], data[train_size:len(data),:]
    y_train, y_test = target[0:train_size], target[train_size:len(data)]



#    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8, random_state=0)

    y_train[y_train==3] = 2
    y_test[y_test==3] = 2

    return (x_train, y_train), (x_test, y_test) 


def load_data_svm(files):
    target, data = load_odds_file_svm(files)
    target = np.array(target)
    data = np.array(data)
 
    '''
    # split into train and test sets
    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    x_train, x_test = data[0:train_size,:], data[train_size:len(data),:]
    y_train, y_test = target[0:train_size], target[train_size:len(data)]
    '''
    x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.85, random_state=0)
    
    y_train[y_train==3] = 2
    y_test[y_test==3] = 2

    return (x_train, y_train), (x_test, y_test)    






        

"""   
#3
tim0=arrow.now()
xdats=pd.read_csv(fxdat,index_col=False,dtype=str,encoding='gb18030')
tn=zt.timNSec('',tim0)
dn=len(xdats.index)
print('#3,xdats tim: {0}s,data num:{1:,} '.format(tn,dn))
"""      
      
      
      




