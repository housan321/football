# -*- coding: utf-8 -*- 
'''
Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2016.12.25 首发
   
Top Football，又称Top Quant for football-简称TFB
TFB极宽足彩量化分析系统，培训课件-配套教学python程序
@ www.TopQuant.vip      www.ziwang.com
QQ总群:124134140   千人大群 zwPython量化&大数据 

  
文件名:tfb_sys.py
默认缩写：import tfb_sys as tfsys
简介：Top极宽量化·足彩系统参数模块
 

'''
#

import sys,os,re
import arrow,bs4,random

import numpy as np
import pandas as pd
import tushare as ts
#import talib as ta

import matplotlib as mpl
from matplotlib import pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
#import multiprocessing
#

#
import numexpr as ne  

#
import zsys
import ztools as zt
import zpd_talib as zta



#-----------global.var&const
#gidDType={'gid':str,'jq':int,'sq':int,'rq':int,'kend':int}
gidNil=['','','','','','',  '-1','-1','0',  '0','-1','-1',  '','','']
gidSgn=['gid','gset','mplay','mtid','gplay','gtid', 'qj','qs','qr',  'kend','kwin','kwinrq', 'tweek','tplay','tsell']
#
poolNil=['','','','','','',  '-1','-1','0',  '0','-1','-1',  '','','', '0',0,0,0, '-9']
poolSgn=['gid','gset','mplay','mtid','gplay','gtid', 'qj','qs','qr',  'kend','kwin','kwinrq', 'tweek','tplay','tsell'
         ,'cid','pwin9','pdraw9','plost9'  , 'kwin_sta']
#

###亚盘参数
###################################################################
pan = {'平手': '0',
       '平手/半球': '0.25',
       '半球': '0.5',
       '半球/一球': '0.75',
       '一球': '1.0',
       '一球/球半': '1.25',
       '球半': '1.5',
       '球半/两球': '1.75',
       '两球': '2.0',
       '两球/两球半': '2.25',
       '两球半': '2.5',
       '两球半/三球':'2.75',
       '三球': '3.0',
       '三球/三球半':'3.25',
       '三球半':'3.5',
       '三球半/四球':'3.75',
       '四球':'4.0',
       '四球/四球半':'4.25',
       '四球半':'4.5',
       '四球半/五球':'4.75',
       '五球':'5.0', 
       '五球/五球半':'5.25',
       '五球半':'5.5',
       '五球半/六球':'5.75',
       '六球':'6.0',
       '六球/六球半':'6.25',
       '六球半':'6.5',
       '六球半/七球':'6.75',
       '七球':'7.0',
       '七球/七球半':'7.25',
       '七球半':'7.5',
       '七球半/八球':'7.75',
       '八球':'8.0',
       
       '受平手/半球': '-0.25',
       '受半球': '-0.5',
       '受半球/一球': '-0.75',
       '受一球': '-1.0',
       '受一球/球半': '-1.25',
       '受球半': '-1.5',
       '受球半/两球': '-1.75',
       '受两球': '-2.0',
       '受两球/两球半': '-2.25',
       '受两球半': '-2.5',
       '受两球半/三球':'-2.75',
       '受三球': '-3.0',
       '受三球/三球半':'-3.25',
       '受三球半':'-3.5',
       '受三球半/四球':'-3.75',
       '受四球':'-4.0',
       '受四球/四球半':'-4.25',
       '受四球半':'-4.5',
       '受四球半/五球':'-4.75',
       '受五球':'-5.0',
       '受五球/五球半':'5.25',
       '受五球半':'5.5',
       '受五球半/六球':'5.75',
       '受六球':'6.0',
       '受六球/六球半':'6.25',
       '受六球半':'6.5',
       '受六球半/七球':'6.75',
       '受七球':'7.0',
       '受七球/七球半':'7.25',
       '受七球半':'7.5',
       '受七球半/八球':'7.75',
       '受八球':'8.0',}  

cidrows= 20


gxdatNil_az=['','','',  0,0,0,0,0,0,0,0, 
         '','','','','', '-1','-1','0','-1','-1', '','' ]

gxdatSgn_az=['gid','cid','cname',
  'mshui0','gshui0','mshui9','gshui9','mrangqui0','grangqui0','mrangqui9','grangqui9',
  #
  'gset','mplay','mtid','gplay','gtid', 
  'qj','qs','qr','kwin','kwinrq',  
  'tweek','tplay']
'''
delSgn_az=['gid','cid','cname',
  #
  'gset','mplay','mtid','gplay','gtid', 
  'qj','qs','qr','kwin','kwinrq',  
  'tweek','tplay']
'''  
usedSgn_az = ['mshui0','gshui0','mshui9','gshui9','mrangqui0','grangqui0','mrangqui9','grangqui9']


'''
delSgn_oz=['cname',
  'vwin0','vdraw0','vlost0','vwin9','vdraw9','vlost9',
  'vback0','vback9',
  'vwin0kali','vdraw0kali','vlost0kali','vwin9kali','vdraw9kali','vlost9kali',
  #
  'gset','mplay','mtid','gplay','gtid', 
  'qj','qs','qr','kwinrq',  
  'tweek','tplay']
'''

usedSgn_oz = ['pwin0','pdraw0','plost0','pwin9','pdraw9','plost9']

###################################################################
###################################################################

###欧盘参数
###################################################################
gxdatNil=['','','',  0,0,0,0,0,0,  0,0,0,0,0,0, 0,0, 0,0,0,0,0,0,
         '','','','','', '-1','-1','0','-1','-1', '','' ]
gxdatSgn=['gid','cid','cname',
  'pwin0','pdraw0','plost0','pwin9','pdraw9','plost9',
  'vwin0','vdraw0','vlost0','vwin9','vdraw9','vlost9',
  'vback0','vback9',
  'vwin0kali','vdraw0kali','vlost0kali','vwin9kali','vdraw9kali','vlost9kali',
  #
  'gset','mplay','mtid','gplay','gtid', 
  'qj','qs','qr','kwin','kwinrq',  
  'tweek','tplay']
###################################################################


###投注参数
###################################################################
gxdatNil_tz=['','','','',
            0,0,0, 0,0,0, 
            0,0,0, 0,0,0,
            0,0,0, 0,0,0, 
            0,0,0, 0,0,0,
            0,0,0, 0,0,0,
            0,0,0, 0,0,0,
            0,0,0, 0,0,0]
gxdatSgn_tz=['gid','mplay','dplay','gplay',
  'odds_m','odds_d','odds_g',
  'probability_m','probability_d','probability_g',
  'beidan_m','beidan_d','beidan_g',
  'betfair_m','betfair_d','betfair_g',
  'price_m','price_d','price_g',
  'volume_m','volume_d','volume_g',
  'profit_m','profit_d','profit_g',
  'bf_idx_m','bf_idx_d','bf_idx_g',
  'hot_idx_m','hot_idx_d','hot_idx_g',
  'profit_idx_m','profit_idx_d','profit_idx_g',
  'profit3','loss3','profit1','loss1','profit0','loss0',
  'hot3','cool3','hot1','cool1','hot0','cool0']

usedSng_tz=['bf_idx_m','bf_idx_d','bf_idx_g',
    'profit3','loss3','profit1','loss1','profit0','loss0',
    'hot3','cool3','hot1','cool1','hot0','cool0']

  
#
retNil=['', 0,0,0,0, 0,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]
retSgn=['xtim', 'kret9','kret3','kret1','kret0',  'knum9','knum3','knum1','knum0',  'ret9','num9','nwin9', 'ret3','ret1','ret0',  'nwin3','nwin1','nwin0',  'num3','num1','num0']
#retNil=[0,0,0,0, 0,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0]
#retSgn=['kret9','kret3','kret1','kret0',  'knum9','knum3','knum1','knum0',  'ret9','num9','nwin9', 'ret3','num3','nwin3', 'ret1','num1','nwin1', 'ret0','num0','nwin0']

#--bt.var  
btvarNil=['', 0,0,0,0, 0,0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0,   0,0,0, 0,0,0,'']
btvarSgn=['xtim', 'kret9','kret3','kret1','kret0',  'knum9','knum3','knum1','knum0',  'ret9','num9','nwin9', 'ret3','ret1','ret0',  'nwin3','nwin1','nwin0',  'num3','num1','num0'
          ,'v1','v2','v3','v4','v5','nday','doc']
  
#self.nsum,self.nwin,self.ndraw,self.nlost=0,0,0,0
#self.kwin,self.kdraw,self.klost=0,0,0
#-------------------
#
#us0='http://trade.500.com/jczq/?date='
#http://odds.500.com/fenxi/shuju-278181.shtml
#http://odds.500.com/fenxi/yazhi-278181.shtml
#http://odds.500.com/fenxi/ouzhi-278181.shtml
us0_gid='http://trade.500.com/jczq/?date='
us0_ext0='http://odds.500.com/fenxi/'
us0_extOuzhi=us0_ext0+'ouzhi-'
us0_extYazhi=us0_ext0+'yazhi-'
us0_extShuju=us0_ext0+'shuju-'
us0_extTouzhu=us0_ext0+'touzhu-'
#
rdat0='/tfbDat/'
rxdat=rdat0+'xdat/'
rmdat=rdat0+'mdat/'
rmlib=rdat0+'mlib/' #ai.mx.lib.xxx

#rgdat=rdat0+'gdat/'
#
rghtm=rdat0+'xhtm/ghtm/'  #gids_htm,days
rhtmOuzhi=rdat0+'xhtm/htm_oz/'
rhtmYazhi=rdat0+'xhtm/htm_az/'
rhtmShuju=rdat0+'xhtm/htm_sj/'
rhtmTouzhu=rdat0+'xhtm/htm_tz/'
#        

#---glibal.lib.xxx
gids=pd.DataFrame(columns=gidSgn,dtype=str)
xdats=pd.DataFrame(columns=gxdatSgn,dtype=str)

gidsFN=''
gidsNum=len(gids.index)
xdatsNum=len(xdats.index)
#
xbars=None
xnday_down=0

#----------class.fbt

class zTopFoolball(object):
    ''' 
    设置TopFoolball项目的各个全局参数
    尽量做到all in one

    '''

    def __init__(self):  
        #----rss.dir
        
        #
        self.tim0Str_gid='2010-01-01'
        self.tim0_gid=arrow.get(self.tim0Str_gid)
        
        #
        self.gid_tim0str,self.gid_tim9str='',''
        self.gid_nday,self.gid_nday_tim9=0,0
        #
        self.tim0,self.tim9,self.tim_now=None,None,None
        self.tim0Str,self.tim9Str,self.timStr_now='','',''
        #
        
        self.kgid=''
        self.kcid=''
        self.ktimStr=''
        #
        #----pool.1day
        self.poolInx=[]
        self.poolDay=pd.DataFrame(columns=poolSgn)
        #----pool.all
        self.poolTrd=pd.DataFrame(columns=poolSgn)
        self.poolRet=pd.DataFrame(columns=retSgn)
        self.poolTrdFN,self.poolRetFN='',''
        #
        self.bars=None
        self.gid10=None
        self.xdat10=None
        
        #
        #--backtest.var
        self.funPre,self.funSta=None,None #funPre()是预处理数，funSta()是分类函数
        self.preVars,self.staVars=[],[]
        #--backtest.ai.var
        #
        self.ai_mxFN0=''
        self.ai_mx_sgn_lst=[]
        self.ai_xlst=[]
        self.ai_ysgn=''
        self.ai_xdat,self.ai_xdat=None,None
        
        #
        #
        
        #
        #--ret.var
        self.ret_nday,self.ret_nWin=0,0
        self.ret_nplay,self.ret_nplayWin=0,0
        
        self.ret_msum=0
        
        

#----------zTopFoolball.init.obj
        

    
