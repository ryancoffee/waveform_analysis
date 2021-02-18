#!/usr/bin/python3

import numpy as np
import lecroyparser 
import h5py
import sys
import os
import re
import math
import joblib
import time

from DataUtils import mypoly
from MyPyClasses.InferenceClasses import SimpleInference


def sigmoid(x,c,w):
    return 1./(1.+np.exp(-(x-c)/w))

def zeroFitRoot(lt,d,thresh=1.):
    tofs = []
    sz = d.shape[0]
    x = []
    y = []
    order = 3
    i = 0
    while i < sz-10:
        while d[i] < thresh:
            i += 1
            if i==sz-10: return tofs
        while d[i+1]>d[i] and i<sz-10:
            i += 1
        while d[i+1]<d[i] and i<sz-10:
            x += [lt[i]]
            y += [d[i]]
            i += 1

        x0 = np.mean(x)
        theta = np.linalg.pinv( mypoly(np.array(x).astype(float),order=order) ).dot(np.array(y).astype(float))
        for j in range(2):
            X0 = np.array([np.power(x0,int(i)) for i in range(order+1)])
            x0 -= theta.dot(X0)/theta.dot([i*X0[(i+1)%(order+1)] for i in range(order+1)])
        tofs += [x0]
        x = []
        y = []
    return tofs

def zeroFit(lt,d,thresh=1.):
    tofs = []
    sz = d.shape[0]
    i = int(2)
    while i < sz-10:
        while d[i] < thresh:
            i += 1
            if i==sz-10: return tofs
        while d[i] > 0:
            i += 1
            if i==sz-10: return tofs
        y = lt[i-2:i+2]
        x = d[i-2:i+2]
        tofs += [ np.linalg.pinv( mypoly(np.array(x).copy().astype(float),order=1) ).dot(np.array(y).astype(float))[0] ]
        i += 2
    return tofs

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


def main():
    vset = 1. #using 1V for vsetting since no retardation has been applied yet
    logvset = math.log(vset)
    modelpath = '/nvme/ave_simulation/SimionSimulationsNM-07-30-2020/backinference_simple25-plate_tune_grid_NM_log40Ret/'
    model = 'simplemodel_logT2logE_2020.09.04.13.39.sav'
    t2e = joblib.load(modelpath + model)

    if len(sys.argv)<2:
        print('syntax is: ./src/load_fromfile.py <datafile>')
        return
    m = re.match('(.+)/(.+)$',sys.argv[1])
    if m:
        path = m.group(1)
        fname = m.group(2)
    else:
        print('Failed the filename match')
        return

    sizeoftrace = getAndreiSize('%s/%s_HEADER.txt'%(path,fname))
    sz = sizeoftrace//8
    ninstances = 100
    nbatches = 200
    times = []
    logtimes = []
    data = []
    FREQ = []
    DDFILT = []
    bwd_dd = 1 # this is in GHz
    bwd_d = 2. # this is in GHz
    bwd = 3. # this is in GHz... very strongly affects the threshold
    histthresh = .01 # .001 I think for the 70V ATI run 08_14_20 and .01 for the 100V ATI run 08_20_20
    LFILT = []
    tstep_ns = 1./40
    nkern = int(1./(tstep_ns * bwd * 2))*2 + 1
    nkern_d = int(1./(tstep_ns * bwd_d * 2))*2 + 1
    nkern_dd = int(1./(tstep_ns * bwd_dd * 2))*2 + 1
    print('kernel lengths:\t%i\t%i\t%i'%(nkern,nkern_d,nkern_dd))
    t0=35
    t0=10
    print('using as tstep = \t%f'%(tstep_ns))
    for batch in range(nbatches):
        print('Processing batch %i of %i instances'%(batch,ninstances))
        raw = np.fromfile('%s/%s'%(path,fname),count=sz*ninstances,offset=batch*ninstances*sizeoftrace,dtype=float)
        data = raw.reshape(ninstances,sz).T
        if batch ==0:
            kern = [math.sin(math.pi*i/nkern) for i in range(nkern)]
            kern_d = [math.sin(2*math.pi*i/nkern_d)*math.sin(math.pi*i/nkern_d) for i in range(nkern_d)] 
            kern_dd = [math.sin(3*math.pi*i/nkern_dd)*math.sin(math.pi*i/nkern_dd) for i in range(nkern_dd)] 
            times = np.array([tstep_ns * i for i in range(data.shape[0])])
            inds = np.where(times>t0+1)
            logtimes = np.zeros(times.shape[0])
            logtimes[inds] = np.log(times[inds]-t0)
            logtimes_mat = np.column_stack([logtimes]*data.shape[1])
            filt = np.zeros(times.shape,dtype=float)
            filt_d = np.zeros(times.shape,dtype=float)
            filt_dd = np.zeros(times.shape,dtype=float)
            filt[:nkern] = kern
            filt_d[:nkern_d] = kern_d
            filt_dd[:nkern_dd] = kern_dd
            FILT = np.tile(np.fft.fft(np.roll(filt,-nkern//2)),(data.shape[1],1)).T
            DFILT = np.tile(np.fft.fft(np.roll(filt_d,-nkern_d//2)),(data.shape[1],1)).T
            DDFILT = np.tile(np.fft.fft(np.roll(filt_dd,-nkern_dd//2)),(data.shape[1],1)).T
            FREQ = np.fft.fftfreq(data.shape[0],tstep_ns) ## in nanoseconds #1./40.) # 1/sampling rate in GHz
                
            b=np.linspace(2,6,2**12+1)
            e=np.linspace(0,2**9,2**12+1)


        DATA = np.fft.fft(data.copy(),axis=0)
        BACK = FILT*DATA
        D = DFILT*DATA
        DD = DDFILT*DATA
        back = np.fft.ifft(BACK,axis=0).real
        deriv = np.fft.ifft(D,axis=0).real
        dderiv = np.fft.ifft(DD,axis=0).real

        y = dderiv*deriv*back*(dderiv>0)
        
        logtlist = []
        for i in range(data.shape[1]):
            logtlist += zeroFitRoot(logtimes,y[:,i],thresh=histthresh)

        if batch == 0:
            histsum = np.histogram(logtlist,bins=b)[0]
        else:
            histsum += np.histogram(logtlist,bins=b)[0]
        np.savetxt('%s/%s.zeroCrossings_fitRoot_histlogtimes_%.1f_%.1f_%.1f_%.3fhistthresh'%(path,fname,bwd,bwd_d,bwd_dd,histthresh),np.column_stack( (b[:-1],histsum) ) )

        ############################# using simple inference #########################
        stime = time.time()
        x = np.c_[logtlist,[logvset]*len(logtlist)]
        X = t2e.pipe0.transform(x)
        Y_0 = t2e.theta0.T.dot(X.T).reshape(-1,1)
        X_res  = t2e.pipe1.transform(np.c_[X[:,1:3].copy(),Y_0])
        Y_res = t2e.theta1.T.dot(X_res.T).reshape(-1,1)
        Y = Y_0 + Y_res
        ens = np.exp(Y)
        print('############ inference model %s took %.3f ms ############'%(model,1e3*(time.time()-stime)))

        if batch == 0:
            ehistsum = np.histogram(ens,bins=e)[0]
        else:
            ehistsum += np.histogram(ens,bins=e)[0]
        np.savetxt('%s/%s.zeroCrossings_fitRoot_histenergies_%.1f_%.1f_%.1f_%.3fhistthresh'%(path,fname,bwd,bwd_d,bwd_dd,histthresh),np.column_stack( (e[:-1],ehistsum) ) )

        
    return
    
if __name__ == "__main__":
    main()
