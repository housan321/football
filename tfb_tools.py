# -*- coding: utf-8 -*- 
'''
Top极宽量化(原zw量化)，Python量化第一品牌 
by Top极宽·量化开源团队 2016.12.25 首发
   
Top Football，又称Top Quant for football-简称TFB
TFB极宽足彩量化分析系统，培训课件-配套教学python程序
@ www.TopQuant.vip      www.ziwang.com
QQ总群:124134140   千人大群 zwPython量化&大数据 
 
文件名:tfb_tools.py
默认缩写：import tfb_tools as tft
简介：Top极宽量化·常用足彩工具函数集

''' 


import os,sys,io,re
import random,arrow,bs4
import numpy as np
import numexpr as ne
import pandas as pd
import tushare as ts
import requests
from bs4 import BeautifulSoup
#
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor,as_completed
#from concurrent.futures import ProcessPoolExecutor
#
#import inspect
#
import zsys
import ztools as zt
import ztools_str as zstr
import ztools_web as zweb
import ztools_data as zdat
#
import tfb_sys as tfsys
import tfb_strategy as tfsty
#
#-----------------------
'''
var&const
tfb.init.obj
tfb.misc
#
tfb.get.dat.xxx
#
tfb.dat.xxx

'''

#-----------------------
#----------var&const

def fb_df_type_xed(df):
    df['qj']=df['qj'].astype(int)
    df['qs']=df['qs'].astype(int)
    df['qr']=df['qr'].astype(int)
    df['kwin']=df['kwin'].astype(int)
    df['kwinrq']=df['kwinrq'].astype(int)

def fb_df_type2float(df,xlst):
    for xsgn in xlst:
        df[xsgn]=df[xsgn].astype(float)

def fb_df_type4mlst(df,nlst,flst):
    for xsgn in nlst:
        df[xsgn]=df[xsgn].astype(int)
        
    for xsgn in flst:
        df[xsgn]=df[xsgn].astype(float)
    
#----------tfb.init.obj
        
def fb_init(rs0='/tfbDat/',fgid=''):
    #1
    xtfb=tfsys.zTopFoolball()
    xtfb.tim_now=arrow.now()
    xtfb.timStr_now=xtfb.tim_now.format('YYYY-MM-DD')
    xtfb.tim0,xtfb.tim0Str=xtfb.tim_now,xtfb.timStr_now
    print('now:',zt.tim_now_str())
    
    #2
    #xtfb.pools=[]
    xtfb.kcid='1'  #官方,3=Bet365
    xtfb.funPre=tfsty.sta00_pre
    xtfb.funSta=tfsty.sta00_sta
    #
    xss=xtfb.timStr_now
    xtfb.poolTrdFN,xtfb.poolRetFN='log\poolTrd_'+xss+'.csv','log\poolRet_'+xss+'.csv'
    #3
    if rs0!='':
        tfsys.rdat=rs0
        tfsys.rxdat=rs0+'xdat/'
        tfsys.rhtmOuzhi=rs0+'xhtm/htm_oz/'
        tfsys.rhtmYazhi=rs0+'xhtm/htm_az/'
        tfsys.rhtmShuju=rs0+'xhtm/htm_sj/'
        tfsys.rhtmTouzhu=rs0+'xhtm/htm_tz/'
        
    #4
    if fgid!='':
        tfsys.gidsFN=fgid
        #xtfb.gids=pd.read_csv(fgid,index_col=0,dtype=str,encoding='gbk')
        tfsys.gids=pd.read_csv(fgid,index_col=False,dtype=str,encoding='gb18030')
        fb_df_type_xed(tfsys.gids)
        tfsys.gidsNum=len(tfsys.gids.index)
        #-----tim.xxx
        xtfb.gid_tim0str,xtfb.gid_tim9str=tfsys.gids['tplay'].min(),tfsys.gids['tplay'].max()
        tim0,tim9=arrow.get(xtfb.gid_tim0str),arrow.get(xtfb.gid_tim9str)
        xtfb.gid_nday,xtfb.gid_nday_tim9=zt.timNDay('',tim0),zt.timNDay('',tim9)
        print('gid tim0: {0}, nday: {1}'.format(xtfb.gid_tim0str,xtfb.gid_nday))    
        print('gid tim9: {0}, nday: {1}'.format(xtfb.gid_tim9str,xtfb.gid_nday_tim9))    
        
        
    #
    return xtfb
        
#----------tfb.misc
def fb_tweekXed(tstr):
    str_week=['星期一','星期二','星期三','星期四','星期五','星期六','星期日']
    str_inx=['1','2','3','4','5','6','0']
    tstr=zstr.str_mxrep(tstr,str_week,str_inx)
    #
    return tstr
            
def fb_kwin4qnum(jq,sq,rq=0):
    if (jq<0)or(sq<0):return -1
    #   
    jqk=jq+rq  #or -rq
    if jqk>sq:kwin=3
    elif jqk<sq:kwin=0
    else:kwin=1
    #
    return kwin

def fb_kwin2pdat(kwin,ds):
    if kwin==3:xd=ds['pwin9']
    elif kwin==1:xd=ds['pdraw9']
    elif kwin==0:xd=ds['plost9']
    #
    return xd    
    
#----------tfb.get.dat.xxx
#def fb_tweekXed(tstr):
            
#def fb_kwin4qnum(jq,sq,rq=0):
    
    
def fb_gid_get4htm(htm):
    bs=BeautifulSoup(htm,'html5lib') # 'lxml'
    df=pd.DataFrame(columns=tfsys.gidSgn,dtype=str)
    ds=pd.Series(tfsys.gidNil,index=tfsys.gidSgn,dtype=str)
    
    #---1#
    zsys.bs_get_ktag_kstr='isend'
    x10=bs.find_all(zweb.bs_get_ktag)
    for xc,x in enumerate(x10):
        #print('\n@x\n',xc,'#',x.attrs)
        ds['gid'],ds['gset']=x['fid'],zstr.str_fltHtmHdr(x['lg'])
        ds['mplay']=zstr.str_fltHtmHdr(x['homesxname'])
        ds['gplay']=zstr.str_fltHtmHdr(x['awaysxname'])
        ds['kend']=x['isend']
        s2=ds['tweek']=x['gdate'].split(' ')[0] #tweek
        ds['tweek']=fb_tweekXed(s2)
        ds['tplay'],ds['tsell']=x['pdate'],x['pendtime']  #tplay,tsell,
        #
        df=df.append(ds.T,ignore_index=True)
        
    #---2#
    x20=bs.find_all('a',class_='score')
    for xc,x in enumerate(x20):
        xss=x['href']
        kss=zstr.str_xmid(xss,'ju-','.sh')
        clst=x.text.split(':')
        #
        ds=df[df['gid']==kss]
        ds=df[df['gid']==kss]
        if len(ds)==1:
            inx=ds.index
            df['qj'][inx]=clst[0]
            df['qs'][inx]=clst[1]
            kwin=fb_kwin4qnum(int(clst[0]),int(clst[1]))
            df['kwin'][inx]=str(kwin)
        
    #---3#
    x20=bs.find_all('td',class_='left_team')
    if (len(x20)==len(x10)):
        for xc,x in enumerate(x20):
            #print('@x',xc,'#',x.a['href'])
            xss=x.a['href']
            if xss.find('/team//')<0:
                xid=zstr.str_xmid(xss,'/team/','/')
                df['mtid'][xc]=xid
                g01=df['gid'][xc]  
                if xid=='':zt.f_addLog('tid-mtid,nil,'+xss+',gid,'+g01)
    #---4#
    x20=bs.find_all('td',class_='right_team')
    if (len(x20)==len(x10)):
        for xc,x in enumerate(x20):
            #print('@x',xc,'#',x.a['href'])
            xss=x.a['href']
            if xss.find('/team//')<0:
                xid=zstr.str_xmid(xss,'/team/','/')
                df['gtid'][xc]=xid
                g01=df['gid'][xc]    
                if xid=='':zt.f_addLog('tid-gtid,nil,'+xss+',gid,'+g01)
    
    #---5#
    df=df[df['gid']!='-1']
    return df
  

def fb_gid_getExt_az4clst(ds,clst):
    i=0;
    ds['mshui0'],ds['gshui0'],ds['mshui9'],ds['gshui9']=clst[i],clst[i+1],clst[i+2],clst[i+3]
    i=i+4;
    ds['mrangqui0'],ds['grangqui0'],ds['mrangqui9'],ds['grangqui9']=clst[i],clst[i+1],clst[i+2],clst[i+3]
    #
    return ds    
    
 
def fb_gid_getExt_oz4clst(ds,clst):
    i=0;
    ds['pwin0'],ds['pdraw0'],ds['plost0']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['pwin9'],ds['pdraw9'],ds['plost9']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['vwin0'],ds['vdraw0'],ds['vlost0']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['vwin9'],ds['vdraw9'],ds['vlost9']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['vback0'],ds['vback9']=clst[i],clst[i+1]
    i=i+2;
    ds['vwin0kali'],ds['vdraw0kali'],ds['vlost0kali']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['vwin9kali'],ds['vdraw9kali'],ds['vlost9kali']=clst[i],clst[i+1],clst[i+2]
    #
    return ds
 
'''
提取投注数据，包括必发指数，庄家盈亏指数，冷热指数
'''    
def fb_gid_getExt_tz4clst(ds,clst):
    p_l = pd.Series(0 ,index=['profit3','loss3','profit1','loss1','profit0','loss0' ],dtype=float)
    h_c = pd.Series(0 ,index=['hot3','cool3','hot1','cool1','hot0','cool0' ],dtype=float)

    i=0;
    ds['mplay'],ds['odds_m'],ds['probability_m']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['beidan_m'],ds['betfair_m'],ds['price_m']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['volume_m'],ds['profit_m'],ds['bf_idx_m']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['hot_idx_m'],ds['profit_idx_m']=clst[i],clst[i+1]
    i=i+2;
    
    ds['dplay'],ds['odds_d'],ds['probability_d']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['beidan_d'],ds['betfair_d'],ds['price_d']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['volume_d'],ds['profit_d'],ds['bf_idx_d']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['hot_idx_d'],ds['profit_idx_d']=clst[i],clst[i+1]
    i=i+2; 
    
    ds['gplay'],ds['odds_g'],ds['probability_g']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['beidan_g'],ds['betfair_g'],ds['price_g']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['volume_g'],ds['profit_g'],ds['bf_idx_g']=clst[i],clst[i+1],clst[i+2]
    i=i+3;
    ds['hot_idx_g'],ds['profit_idx_g']=clst[i],clst[i+1]
    #
    usedSng_tz = ['bf_idx_m','bf_idx_d','bf_idx_g',
                  'hot_idx_m','hot_idx_d','hot_idx_g',
                  'profit_idx_m','profit_idx_d','profit_idx_g']
    
    ds[ds[tfsys.gxdatSgn_tz]=='-'] = 0  #有些数据是 '-',把它定义为0
    ds[tfsys.usedSng_tz] = ds[tfsys.usedSng_tz].astype(float) #把相关的字符串转数字
    ds[usedSng_tz] = ds[usedSng_tz].astype(float)
    if ds['hot_idx_m'] >= 0:
        h_c['hot3'] = ds['hot_idx_m']
    else: h_c['cool3'] = -ds['hot_idx_m']
    if ds['hot_idx_d'] >= 0:
        h_c['hot1'] = ds['hot_idx_d']
    else: h_c['cool1'] = -ds['hot_idx_d']
    if ds['hot_idx_g'] >= 0:
        h_c['hot0'] = ds['hot_idx_g']
    else: h_c['cool0'] = -ds['hot_idx_g']

    if ds['profit_idx_m'] >= 0:
        p_l['profit3'] = ds['profit_idx_m']
    else: p_l['loss3'] = -ds['profit_idx_m']
    if ds['profit_idx_d'] >= 0:
        p_l['profit1'] = ds['profit_idx_d']
    else: p_l['loss1'] = -ds['profit_idx_d']
    if ds['profit_idx_g'] >= 0:
        p_l['profit0'] = ds['profit_idx_g']
    else: p_l['loss0'] = -ds['profit_idx_g']

    ds['profit3'],ds['profit1'],ds['profit0'] = p_l['profit3'],p_l['profit1'],p_l['profit0']
    ds['loss3'],ds['loss1'],ds['loss0'] = p_l['loss3'],p_l['loss1'],p_l['loss0']

    ds['hot3'],ds['hot1'],ds['hot0'] = h_c['hot3'],h_c['hot1'],h_c['hot0']
    ds['cool3'],ds['cool1'],ds['cool0'] = h_c['cool3'],h_c['cool1'],h_c['cool0']

    
    return ds    


  
def fb_gid_getExt_oz4htm(htm,bars,ftg=''):
    bs=BeautifulSoup(htm,'html5lib') # 'lxml'
    x10=bs.find_all('tr',ttl='zy')
    df=pd.DataFrame(columns=tfsys.gxdatSgn)
    ds=pd.Series(tfsys.gxdatNil,index=tfsys.gxdatSgn)
    xc,gid=0,bars['gid']
    xlst=['gset','mplay','mtid','gplay','gtid', 'qj','qs','qr','kwin','kwinrq','tplay','tweek']
    for xc,x in enumerate(x10):
        #print('\n@x\n',xc,'#',x.attrs)
        x2=x.find('td',class_='tb_plgs');#print(x2.attrs)
        ds['gid'],ds['cid'],ds['cname']=gid,x['id'],x2['title']
        #
        x20=x.find_all('table',class_='pl_table_data');
        clst=zt.lst4objs_txt(x20,['\n','\t','%'])
        ds=fb_gid_getExt_oz4clst(ds,clst)
        #
        zdat.df_2ds8xlst(bars,ds,xlst)
        df=df.append(ds.T,ignore_index=True)
    
    #
    #print('xx',xc)
    #--footer
    if xc>0:
        x10=bs.find_all('tr',xls='footer')
        
        for xc,x in enumerate(x10):
            #print('\n@x\n',xc,'#',x.attrs)
            if xc<3:
                x20=x.find_all('table',class_='pl_table_data');
                clst=zt.lst4objs_txt(x20,['\n','\t','%'])
                ds['gid']=gid
                if xc==0:ds['cid'],ds['cname']='90005','gavg'
                if xc==1:ds['cid'],ds['cname']='90009','gmax'
                if xc==2:ds['cid'],ds['cname']='90001','gmin'
                #
                zdat.df_2ds8xlst(bars,ds,xlst)
                ds=fb_gid_getExt_oz4clst(ds,clst)
                #
                df=df.append(ds.T,ignore_index=True)
        #
        if ftg!='':df.to_csv(ftg,index=False,encoding='gb18030')
    #
    return df
 
##########################################################################
########################################################################## 
def lst4objs_txt_az(xobjs,fltLst=[]):
    clst=[]
    odds = [0] * 8
    for x in xobjs:
        #css=x.text.replace('\n','')
        css=zstr.str_flt(x.get_text(),fltLst)
        c20=css.split(' ')    
        for c in c20:
            if c!='' and c!='升' and c!='降':
                clst.append(c)
    cl = clst[0:3]+clst[-3:]
    odds[0] = cl[0]
    odds[1] = cl[2]
    odds[2] = cl[3]
    odds[3] = cl[5]
    begin_pan = float(tfsys.pan[cl[1]])
    end_pan = float(tfsys.pan[cl[4]])
    if begin_pan >= 0:
        odds[5] =  begin_pan        #  1- begin_pan * 0.125      #乘以0.125是为了归一化
    else: odds[4] =  -begin_pan      # 1- begin_pan * -0.125
    if end_pan >= 0:
        odds[7] =  end_pan         # 1- end_pan * 0.125
    else: odds[6] = -end_pan       # 1- end_pan * -0.125
        
    return odds  


def fb_gid_getExt_az4htm(htm,bars,ftg=''):  
    bs=BeautifulSoup(htm,'html5lib') # 'lxml'
    x10=bs.find_all('tr',xls='row')
    df=pd.DataFrame(columns=tfsys.gxdatSgn_az)
    ds=pd.Series(tfsys.gxdatNil_az,index=tfsys.gxdatSgn_az)
    xc,gid=0,bars['gid']

    xlst=['gset','mplay','mtid','gplay','gtid', 'qj','qs','qr','kwin','kwinrq','tplay','tweek']
    for xc,x in enumerate(x10):
        #print('\n@x\n',xc,'#',x.attrs)
        x2=x.find('td',class_='tb_plgs');#print(x2.attrs)
        ds['gid'],ds['cid']=gid,x['id']
        cname = x2.get_text()
        cname = cname[0: int(len(cname)/2)]
        ds['cname'] = cname
        #
        x20=x.find_all('table',class_='pl_table_data');
        clst=lst4objs_txt_az(x20,['\n','\t','%', '↓', '↑'])
        ds=fb_gid_getExt_az4clst(ds,clst)
        #
        zdat.df_2ds8xlst(bars,ds,xlst)
        df=df.append(ds.T,ignore_index=True)

    if ftg!='':df.to_csv(ftg,index=False,encoding='gb18030')

    return df    

  
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

################################################################################## 
##################################################################################
    
def fb_gid_getExt010(x10):
    bars=pd.Series(x10,index=tfsys.gidSgn,dtype=str)
    gid=bars['gid']
    #
    fss_az=tfsys.rhtmYazhi+gid+'_az.htm'
    uss_az=tfsys.us0_extYazhi+gid+'.shtml'
    fss_oz=tfsys.rhtmOuzhi+gid+'_oz.htm'
    uss_oz=tfsys.us0_extOuzhi+gid+'.shtml' 
    fss_tz=tfsys.rhtmTouzhu+gid+'_tz.htm'
    uss_tz=tfsys.us0_extTouzhu+gid+'.shtml'
    
    htm_az=zweb.web_get001txtFg(uss_az,fss_az)
    htm_oz=zweb.web_get001txtFg(uss_oz,fss_oz) #zt.zt_web_get001txtFg or(fsiz<5000):
    htm_tz=zweb.web_get001txtFg(uss_tz,fss_tz)
    #  
    fxdat_az=tfsys.rxdat+gid+'_az.dat'
    fxdat_oz=tfsys.rxdat+gid+'_oz.dat'
    fxdat_tz=tfsys.rxdat+gid+'_tz.dat'
    fsiz_az=zt.f_size(fxdat_az);#print(zsys.sgnSP4,'@',fsiz,fxdat)
    fsiz_oz=zt.f_size(fxdat_oz);
    fsiz_tz=zt.f_size(fxdat_tz);
    #
    #print('xtfb.bars',xtfb.bars)
    if (fsiz_az<1000)or(tfsys.xnday_down<10): 
        fb_gid_getExt_az4htm(htm_az,bars,ftg=fxdat_az)
        
    if (fsiz_oz<1000)or(tfsys.xnday_down<10): 
        fb_gid_getExt_oz4htm(htm_oz,bars,ftg=fxdat_oz)

    if (fsiz_tz<1000)or(tfsys.xnday_down<10): 
        fb_gid_getExt_tz4htm(htm_tz,bars,ftg=fxdat_tz)
 
    
    return fxdat_az,fxdat_oz,fxdat_tz

    
def fb_gid_getExt(df):
    dn9=len(df['gid'])
    for i, row in df.iterrows():
        #xtfb.kgid=row['gid']
        #xtfb.bars=row
        fb_gid_getExt010(row.values)
        #
        print(zsys.sgnSP8,i,'/',dn9,'@ext')

    
def fb_gid_getExtPool(df,nsub=5):
    pool=ThreadPoolExecutor(max_workers = nsub)
    xsubs = [pool.submit(fb_gid_getExt010,x10) for x10 in df.values]
    #
    dn9=len(df['gid'])
    ns9=str(dn9)
    for xsub in as_completed(xsubs):
        fss=xsub.result(timeout=20);
        print('@_getExtPool,xn9:',ns9,fss)
        
    
def fb_gid_get_nday(xtfb,timStr,fgExt=False):
    if timStr=='':ktim=xtfb.tim_now
    else:ktim=arrow.get(timStr)
    #
    nday=tfsys.xnday_down
    for tc in range(0, nday):
        xtim=ktim.shift(days= -tc)
        xtimStr=xtim.format('YYYY-MM-DD')
        #print('\nxtim',xtim,xtim<xtfb.tim0_gid)
        #
        xss=str(tc)+'#,'+xtimStr+',@'+ zt.get_fun_nam()
        zt.f_addLog(xss)
        if xtim<xtfb.tim0_gid:
            print('#brk;')
            break
        #
        
        fss=tfsys.rghtm+xtimStr+'.htm'
        uss=tfsys.us0_gid+xtimStr
        print(timStr,tc,'# update--',fss)
        #
        htm=zweb.web_get001txtFg(uss,fss)
        if len(htm)>5000:
            df=fb_gid_get4htm(htm)
            if len(df['gid'])>0:
                tfsys.gids=tfsys.gids.append(df)
                tfsys.gids.drop_duplicates(subset='gid', keep='last', inplace=True)
                #
                #if fgExt:fb_gid_getExt(df)
                if fgExt:fb_gid_getExtPool(df)
    #
    if tfsys.gidsFN!='':
        print('+++++')
        print(tfsys.gids.tail())
        tfsys.gids.to_csv(tfsys.gidsFN,index=False,encoding='gb18030')

#-----tfb.dat.xxx
def fb_xdat_xrd020(fsr,xlst,ysgn='kwin',k0=1,fgPr=False):    
    
    #1
    df=pd.read_csv(fsr,index_col=False,encoding='gb18030')
    #2
    if ysgn=='kwin':
        df[ysgn]=df[ysgn].astype(str)
        df[ysgn].replace('3','2', inplace=True)
        #df['kwin'].replace('3','2', inplace=True)
    #3
    df[ysgn]=df[ysgn].astype(float)
    df[ysgn]=round(df[ysgn]*k0).astype(int)              
    #4              
    x_dat,y_dat= df[xlst],df[ysgn]   
      
    #5
    if fgPr:
        print('\n',fsr);
        print('\nx_dat');print(x_dat.tail())
        print('\ny_dat');print(y_dat.tail())
        #df.to_csv('tmp\df.csv',index=False,encoding='gb18030')
    #6
    return  x_dat,y_dat     
        
def fb_xdat_xlnk(rs0,ftg):
    flst=zt.lst4dir(rs0)
    df9=pd.DataFrame(columns=tfsys.gxdatSgn,dtype=str)
    for xc,fs0 in enumerate(flst):
        fss=rs0+fs0
        print(xc,fss)
        df=pd.read_csv(fss,index_col=False,dtype=str,encoding='gb18030')
        #
        df2=df[df['kwin']!='-1']
        df9=df9.append(df2,ignore_index=True)
        #
        if (xc % 2000)==0:
            #df9.to_csv(ftg,index=False,encoding='gb18030')
            fs2='tmp/x_'+str(xc)+'.dat';print(fs2,fss)
            df9.to_csv(fs2,index=False,encoding='gb18030')
    #
    df9.to_csv(ftg,index=False,encoding='gb18030')        
