#!/usr/bin/python3

import numpy as np
import lecroyparser 
import h5py
import sys
import os
import re
import math

from DataUtils import mypoly

def zeroCrossings2energy(data,thresh,tstep = 1.,t0 = 50.):
    # from fitting Ave simulations with theta0 = memmap([ 6.39445393, -0.03775023, -0.46237318])
    # this is log(t - t0[ns]) = [6.39445393, -0.03775023, -0.46237318].T dot [1,ln(vret[V]),ln(ekin[eV])]
    #                                                       
    # fit f(x) energiesfile u (log($1-50)):(log(1.55*($0+1)+30)) via a,b,c,d
    # f(x)= a + b*x + c* x**2 + d* x**3
    # Final set of parameters            Asymptotic Standard Error
    # =======================            ==========================
    # a               = -99.8063         +/- 10.31        (10.33%)
    # b               = 72.8784          +/- 6.79         (9.317%)
    # c               = -16.7728         +/- 1.491        (8.886%)
    # d               = 1.26497          +/- 0.109        (8.616%)
    # new fitting using also the 20Vacc results
    # Final set of parameters            Asymptotic Standard Error
    # =======================            ==========================
    # a               = 13.0505          +/- 0.259        (1.985%)
    # b               = -2.96445         +/- 0.1147       (3.869%)
    # c               = 0.199342         +/- 0.01267      (6.354%)
    # 
    # better points from data_fs/energies.notes file (based on 200 threshold for 20Vacc case, still using 1000 threshold for 30Vacc case
    # Final set of parameters            Asymptotic Standard Error
    # =======================            ==========================
    # a               = 12.6565          +/- 0.2341       (1.85%)
    # b               = -2.79381         +/- 0.1044       (3.737%)
    # c               = 0.18091          +/- 0.01161      (6.417%)


    c = np.array([12.6565,-2.79381,0.18091])
    tofs = zeroCrossings(data,thresh,tstep)
    logt = [math.log(v-t0) for v in tofs if (v>t0)]
    X_f = np.row_stack((np.ones(len(logt)),logt,np.power(logt,2)))
    e = list(np.exp(c.dot(X_f)))
    return tofs,e

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
        for j in range(order+1):
            X0 = np.array([np.power(x0,int(i)) for i in range(order+1)])
            x0 -= theta.dot(X0)/theta.dot([i*X0[(i+1)%(order+1)] for i in range(order+1)]) # this seems like maybe it should be wrong
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

def zeroCrossingTimes(times,data,thresh,tstep = 1.):
    tofs = []
    i = int(0)
    sz = data.shape[0]
    while i < data.shape[0]-1:
        while data[i] < thresh:
            i += 1
            if i == sz-1: break
        if i == sz-1: break
        while data[i] > 0:
            i += 1
            if i == sz-1: break
        tofs = tofs + [ (times[i]-times[i-1])/float(data[i]-data[i-1])*(-data[i-1]) + times[i-1] ]
        if i == sz-1: break
        while data[i] < 0:
            i += 1
            if i == sz-1: break
    return tofs 
def zeroCrossings(ltimes,data,thresh,tstep = 1.):
    tofs = []
    i = int(0)
    sz = data.shape[0]
    while i < data.shape[0]-1:
        while data[i] < thresh:
            i += 1
            if i == sz-1: break
        if i == sz-1: break
        while data[i] > 0:
            i += 1
            if i == sz-1: break
        tofs = tofs + [ (ltimes[i]-ltimes[i-1])/float(data[i]-data[i-1])*(-data[i-1]) + ltimes[i-1] ]
        if i == sz-1: break
        while data[i] < 0:
            i += 1
            if i == sz-1: break
    return tofs 

def getHeaderBytesTimes(fname):
    data = lecroyparser.ScopeData(fname)
    nvals = len(data.y)
    f = open(fname,'br')
    buf = f.read()
    f.close()
    nseek = int(len(buf) - 2*nvals)
    return (nseek,nvals,data.x)

def loadArrayBytes_int8(fname,nseek,nvals):
    f = open(fname,'br')
    f.seek(nseek)
    buf = f.read()
    f.close()
    return np.array([np.int8(v) for v in buf],dtype=np.int8)

def loadArrayBytes_int16(fname,nseek,nvals,byteorder = 'little'):
    f = open(fname,'br')
    f.seek(nseek)
    data = []
    for i in range(nvals):
        data.append(int.from_bytes(f.read(2),byteorder=byteorder))
    f.close()
    return np.array(data,dtype=np.int16)

def getWeinerFilter(data,FREQ,cut = 2.0,noise = 0.1):
    W = np.zeros(data.shape[0],dtype=float)
    inds = np.where(np.abs(FREQ)<cut)
    c2 = 0.5*(1.+np.cos(FREQ[inds]*np.pi/cut))
    W[inds] = c2 / (c2 + noise)
    return W

def getacFilter(data,FREQ,cut = 0.2):
    AC = np.ones(data.shape[0],dtype=float)
    inds = np.where(np.abs(FREQ)<cut)
    AC[inds] = 0.5*(1. - np.cos(FREQ[inds]*np.pi/cut))
    return AC

def checktracelength(raw):
    ntest = raw.shape[0]//20
    Y = np.abs(np.fft.fft(raw[:ntest]))
    F = np.fft.fftfreq(ntest)
    i = np.argmax(Y[10:len(Y)//4])
    n = int((1./F[i] + 500)/1e3)*1000 + 2
    instances = raw.shape[0]//n
    print('how do you like as the sample intervals:\t%i'%n)
    return (n,instances)

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

def logTlogE(x,x0=3.6,theta = [4.39,-0.72]):
#Final set of parameters            Asymptotic Standard Error
# x0=3.6
#=======================            ==========================
#a               = 4.39133          +/- 0.001283     (0.02922%)
#b               = -0.72            +/- 0.01098      (1.525%)

#correlation matrix of the fit parameters:
#                a      b      
#a               1.000 
#b               0.266  1.000 
    X = np.array([1.,x-x0])
    return np.array(theta).dot(X)

def logTlogE_jac(x):
    return x

def main():
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
                
            #b=np.linspace(60,90,2**10+1)
            #b=np.linspace(0.,10,2**12+1)
            b=np.linspace(2,6,2**12+1)
            ebins=np.array([math.exp(logTlogE(v)) for v in b[:-1]])

        if batch ==0:
            sumsig = np.sum(data,axis=1)
            np.savetxt('%s/%s.samplesig'%(path,fname),np.column_stack((times,data)))
        else:
            sumsig += np.sum(data,axis=1)
        
        np.savetxt('%s/%s.sumsig'%(path,fname),np.column_stack((times,sumsig)))
    

        DATA = np.fft.fft(data.copy(),axis=0)
        BACK = FILT*DATA
        D = DFILT*DATA
        DD = DDFILT*DATA
        back = np.fft.ifft(BACK,axis=0).real
        deriv = np.fft.ifft(D,axis=0).real
        dderiv = np.fft.ifft(DD,axis=0).real
        #derivscale = np.max(np.abs(deriv.copy()))
        #dderivscale = np.max(np.abs(dderiv.copy()))
        #datascale = np.max(np.abs(data.copy())) 

        y = dderiv*deriv*back*(dderiv>0)
        
        if batch%50==0:
            np.savetxt('%s/%s.%i.dfft'%(path,fname,batch),np.column_stack( (FREQ,np.abs(D)) ))
            np.savetxt('%s/%s.%i.ddfft'%(path,fname,batch),np.column_stack( (FREQ,np.abs(DD)) ))
            np.savetxt('%s/%s.%i.back'%(path,fname,batch),np.column_stack( (times,back) ))
            np.savetxt('%s/%s.%i.dback'%(path,fname,batch),np.column_stack( (times,deriv) ))
            np.savetxt('%s/%s.%i.ddback'%(path,fname,batch),np.column_stack( (times,dderiv) ))
            np.savetxt('%s/%s.%i.dlogic'%(path,fname,batch),np.column_stack( (logtimes,y) ))
        logtlist = []
        for i in range(data.shape[1]):
            #logtlist += zeroCrossingTimes(times,y[:,i],thresh=histthresh)
            #logtlist += zeroCrossings(logtimes,y[:,i],thresh=histthresh)
            #logtlist += zeroFit(logtimes,y[:,i],thresh=histthresh)
            logtlist += zeroFitRoot(logtimes,y[:,i],thresh=histthresh)

        if batch == 0:
            histsum = np.histogram(logtlist,bins=b)[0]
            #histsum = np.sum(hmat,axis=1)
        else:
            histsum += np.histogram(logtlist,bins=b)[0]
            #histsum += np.sum(hmat,axis=1)
        np.savetxt('%s/%s.zeroCrossings_fitRoot_histlogtimes_%.1f_%.1f_%.1f_%.3fhistthresh'%(path,fname,bwd,bwd_d,bwd_dd,histthresh),np.column_stack( (b[:-1],ebins,histsum) ) )
        
    return
    
if __name__ == "__main__":
    main()
