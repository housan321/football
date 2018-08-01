#coding=utf-8
'''
Created on 2016.12.25
Top Football
Top Quant for football-极宽足彩量化分析系统
简称TFB，培训课件-配套教学python程序
@ www.TopQuant.vip      www.ziwang.com

'''

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
#
#-----------------------

  
def fb_gid_getExt_tz4clst(ds,clst):
    i=0;
    ds['mplay'],ds['odds_m'],ds['probability_m']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['beidan_m'],ds['betfair_m'],ds['price_m']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['volume_m'],ds['profit_m'],ds['betfair_idx_m']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['hot_idx_m'],ds['profit_idx_m']=clst[i],clst[i+1]
    i=i+2;
    
    ds['dplay'],ds['odds_d'],ds['probability_d']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['beidan_d'],ds['betfair_d'],ds['price_d']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['volume_d'],ds['profit_d'],ds['betfair_idx_d']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['hot_idx_d'],ds['profit_idx_d']=clst[i],clst[i+1]
    i=i+2; 
    
    ds['gplay'],ds['odds_g'],ds['probability_g']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['beidan_g'],ds['betfair_g'],ds['price_g']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['volume_g'],ds['profit_g'],ds['betfair_idx_g']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['hot_idx_g'],ds['profit_idx_g']=clst[i],clst[i+1]
    #
    return ds





  
def fb_gid_getExt_tz4htm(htm,bars,ftg=''):
    bs=BeautifulSoup(htm,'html5lib') # 'lxml'
    x10=bs.find_all('div',class_='M_box record')
    df=pd.DataFrame(columns=tfsys.gxdatSgn_tz)
    ds=pd.Series(tfsys.gxdatNil_tz,index=tfsys.gxdatSgn_tz)
    ds['gid'] = bars['gid']
    
    if len(x10) > 0:
        x = x10[0]
        #
        x20=x.find_all('tr')
        if len(x20) >= 5:
            clst=zt.lst4objs_txt(x20[2:5],['\n','\t','%'])
            ds=fb_gid_getExt_tz4clst(ds,clst) 
            df=df.append(ds.T,ignore_index=True)
            if ftg!='':df.to_csv(ftg,index=False,encoding='gb18030')
    #
    return df
          
    
#-----------------------    

#1
gid='668003'
fgid='/tfbDat/gid2018.dat'
gids=pd.read_csv(fgid,index_col=False,dtype=str,encoding='gb18030')

#2
g10=gids[gids['gid']==gid]

bars=pd.Series(list(g10.values[0]),index=list(g10))
print('\n#2')
print(bars)
print('\ntype(g10),',type(g10))

#3
fhtm,ftg='dat/'+gid+'_tz.htm','tmp/'+gid+'_tz.dat'
htm=zt.f_rd(fhtm)
df=fb_gid_getExt_tz4htm(htm,bars,ftg)
print('\n#3')
print(df.tail())

#-----------------------
print('\nok!')
