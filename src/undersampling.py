#!/usr/bin/python3

import numpy as np
from scipy import fft
import re
import sys
import time
from threading import get_ident
from tqdm import tqdm
import multiprocessing
from joblib import Parallel,delayed

from DataUtils import mypoly

def getAndreiSize(fname):
    f = open(fname,'r')
    line = f.readline().strip()
    print(line)
    m = re.search(':\s+(\d+)$',line)
    if m:
        return int(m.group(1))
    print('failed match')
    f.close()
    return 8

def scanedges(d,minthresh):
    tofs = []
    sz = d.shape[0]
    order = 3
    i = 1
    while i < sz-10:
        while d[i] > minthresh:
            i += 1
            if i==sz-10: return tofs
        while i<sz-10 and d[i]<d[i-1]:
            i += 1
        start = i
        i += 1
        while i<sz-10 and d[i]>d[i-1]:
            i += 1
        stop = i
        if stop-start<4:
            continue
        x = np.arange(stop-start,dtype=float)
        y = d[start:stop]
        x0 = float(stop)/2.
        y -= (y[0]+y[-1])/2.
        theta = np.linalg.pinv( mypoly(np.array(x).astype(float),order=order) ).dot(np.array(y).astype(float))
        for j in range(3): # 3 rounds of Newton-Raphson
            X0 = np.array([np.power(x0,int(i)) for i in range(order+1)])
            x0 -= theta.dot(X0)/theta.dot([i*X0[(i+1)%(order+1)] for i in range(order+1)]) # this seems like maybe it should be wrong
        tofs += [start + x0]
    return tofs

def findedges(mat,minthresh):
    order = int(3)
    edges = [[] for i in range(mat.shape[1])]
    sz = mat.shape[0]
    for i in range(mat.shape[1]):
        edges[i] = scanedges(mat[:,i],minthresh)
    return edges

def pairedges(edges1,edges2):
    if len(edges2)>len(edges1):
        a,b = edges1,edges2
    else:
        a,b = edges2,edges1
    res = []
    for e in a:
        i = np.argmin(np.abs(b-e))
        res += [[e,b[i]]]
    return res




def run_upscale(upscale,params):
    out = []
    path = params['path']
    fname = params['fname']
    sz = params['sz']
    sizeoftrace = params['sizeoftrace']
    nbatches = params['nbatches']
    ninstances = params['ninstances']
    tstep_ns = 1./40
    #thresh = hard to catch for upscale=1; -40 for upscale=2; -20 for upscale=3; -10 for upscale=4;-5 for upscale=6;-2.5 for upscale=8
    tstep_under_ns = tstep_ns*6/upscale ## acutal Abaco sampling will be 6 GSps, but for this we simply take every 6th step of the waveform.

    upthresh = {1:-40,2:-40,3:-20,4:-10,6:-5,8:-2.5,10:-1.5}

    for batch in range(nbatches):
        print('Processing batch %i of %i instances for upscale %.2f'%(batch,ninstances,upscale))
        raw = np.fromfile('%s/%s'%(path,fname),count=sz*ninstances,offset=batch*ninstances*sizeoftrace,dtype=float)
        data = 1e3*raw.reshape(ninstances,sz).T # data in millivolts
        d = np.row_stack((data,np.flipud(data)))
        d_ = np.row_stack((data[::6,:],np.flipud(data[::6,:])))
        frq = np.arange(d.shape[0],dtype=float)
        flt = (1.+np.cos(frq*np.pi/frq.shape[0]))
        filt = np.tile(flt,(d.shape[1],1)).T
        frq_ = np.arange(d_.shape[0],dtype=float)
        flt_ = (1.+np.cos(frq_*np.pi/frq_.shape[0]))
        filt_ = np.tile(flt_,(d_.shape[1],1)).T
        DC = fft.dct(d,axis=0)
        DC_ = fft.dct(d_,axis=0)
        DC[frq.shape[0]:,:] = 0
        DC[:frq.shape[0],:] *= filt 
        DC_[frq_.shape[0]:,:] = 0
        DC_[:frq_.shape[0],:] *= filt_

        if batch%10==0: np.savetxt('%s/processed/%s_b%i_sample.dat'%(path,fname,batch),d,fmt='%.3f')
        DC_up = np.zeros((DC_.shape[0]*upscale,DC_.shape[1]),dtype=float)
        DC_up[:DC_.shape[0],:] = DC_
        dcc = fft.idct(DC,axis=0)
        dsc = fft.idst(DC,axis=0)
        #dcc_ = fft.idct(DC_,axis=0)
        #dsc_ = fft.idst(DC_,axis=0)
        dcc_up = fft.idct(DC_up,axis=0)
        dsc_up = fft.idst(DC_up,axis=0)
        logic = (dsc*dcc)[:DC.shape[0]//2,:]
        #logic_ = (dsc_*dcc_)[:frq_.shape[0]//2,:]
        logic_up = (dsc_up*dcc_up)[:DC_.shape[0]//2*upscale,:]
        if batch%10==0:
            np.savetxt('%s/processed/%s_b%i_dct.dat'%(path,fname,batch),DC,fmt='%.3f')
            np.savetxt('%s/processed/%s_b%i_idct.dat'%(path,fname,batch),dcc,fmt='%.3f')
            np.savetxt('%s/processed/%s_b%i_idst.dat'%(path,fname,batch),dsc,fmt='%.3f')
            np.savetxt('%s/processed/%s_b%i_logic.dat'%(path,fname,batch),logic,fmt='%.3f')
            #np.savetxt('%s/processed/%s_b%i_logic_.dat'%(path,fname,batch),logic_,fmt='%.3f')
            np.savetxt('%s/processed/%s_b%i_logic_up.dat'%(path,fname,batch),logic_up,fmt='%.3f')
        f = open('%s/processed/%s_b%i_logic_edges.dat'%(path,fname,batch),'w')
        f_up = open('%s/processed/%s_b%i_logic_up_edges.dat'%(path,fname,batch),'w')
        for i in range(ninstances):
            logic_edges = scanedges(logic[:,i],-100)
            line = '\t'.join(['%.3f'%(e) for e in logic_edges]) + '\n'
            f.write(line)
            #print(len(logic_edges),line)
            logic_up_edges = scanedges(logic_up[:,i],upthresh[upscale])
            line = '\t'.join(['%.3f'%(4*e/6.) for e in logic_up_edges]) + '\n'
            f_up.write(line)
            out += pairedges([tstep_ns*e for e in logic_edges]
                    ,[tstep_under_ns*e for e in logic_up_edges])
        f.close()
        f_up.close()
        if batch%10==0: 
            np.savetxt('%s/processed/%s_logic_compare.out'%(path,fname),np.column_stack(out),fmt='%.3f')
            f = open('%s/processed/%s_logic_compare_upscale%i.dat'%(path,fname,upscale),'w')
            _ = [f.write('%.4f\t%.4f\n'%(p[0],p[1])) for p in out]
            f.close()
            hbins = np.arange(-10,10,0.02)
            h = np.histogram(np.array(out)[:,1]-np.array(out)[:,0],hbins)[0]
            np.savetxt('%s/processed/%s_logic_compare_upscale%i.hist'%(path,fname,upscale),np.column_stack((hbins[:-1],h)),fmt='%.3f')
    hbins = np.arange(-10,10,0.02)
    h = np.histogram(np.array(out)[:,1]-np.array(out)[:,0],hbins)[0]
    np.savetxt('%s/processed/%s_logic_compare_upscale%i.hist'%(path,fname,upscale),np.column_stack((hbins[:-1],h)),fmt='%.3f')

    return





def main():

    path = ''
    fname = ''
    if len(sys.argv)<2:
        print('syntax is: %s <datafile>'%sys.argv[0])
        return
    m = re.match('^(.+)/(.+)$',sys.argv[1])
    if m:
        path = m.group(1)
        fname = m.group(2)
    else:
        print('Failed the filename match')
        return


    sizeoftrace = getAndreiSize('%s/%s_HEADER.txt'%(path,fname))
    sz = sizeoftrace//8
    ninstances = 64
    nbatches = 64 


    num_cores = multiprocessing.cpu_count()
    print(num_cores)
    upscalelist = [1,2,3,4,6,8,10]
    #upscalelist = tqdm([1,2,3,4,6,8,10])

    params = {  'path':path,
                'fname':fname,
                'sz':sz,
                'sizeoftrace':sizeoftrace,
                'nbatches':nbatches,
                'ninstances':ninstances,
                }

    _ = Parallel(n_jobs=num_cores, require='sharedmem')(delayed(run_upscale)(upscale,params) for upscale in upscalelist)
    #_ = [run_upscale(upscale,params) for upscale in upscalelist]

    return

if __name__ == '__main__':
    main()
