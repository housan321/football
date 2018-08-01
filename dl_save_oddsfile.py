# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 17:53:49 2018

@author: Administrator
"""


import dl_data_process as dp


odds_file_name = 'f_odds(more).dat'


dp.save_odds_to_file('odds2018(svm1).dat', [0, 10000])
dp.save_odds_to_file('odds2018(svm2).dat', [10000, 20000]) 




'''
# kwin, data = dp.load_odds_file2(['f_odds(more)1.dat', 'f_odds(more)2.dat', 'f_odds(more)3.dat'])

files = ['f_odds(more)1.dat', 'f_odds(more)2.dat', 'f_odds(more)3.dat']
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = dp.load_data2(files)
'''
