#!/usr/bin/python3

import numpy as np
import lecroyparser 
import h5py
import sys
import os
import re
import math

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

def zeroCrossings(data,thresh,tstep = 1.):
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
        tofs = tofs + [(1./float(data[i]-data[i-1])*data[i] + float(i))*tstep]
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
    dd_bwd = 8. # this is in GHz
    histthresh = .4 # .4 seems the best for up to 100eV electrons, only %.1f SHould make the histthresh also a function of the log(tof)
    # try building a histogram of peak hights versus log(tof) and use that to decide on the threshold functon
    logic_bwd=8
    LFILT = []
    tstep_ns = 1./40
    t0=35
    print('using as tstep = \t%f'%(tstep_ns))
    for batch in range(nbatches):
        print('Processing batch %i of %i instances'%(batch,ninstances))
        raw = np.fromfile('%s/%s'%(path,fname),count=sz*ninstances,offset=batch*ninstances*sizeoftrace,dtype=float)
        data = raw.reshape(ninstances,sz).T
        if batch ==0:
            times = np.array([tstep_ns * i for i in range(data.shape[0])])
            logtimes = np.column_stack([np.log(times-t0 + 1j*tstep_ns).real]*data.shape[1])
            FREQ = np.fft.fftfreq(data.shape[0],tstep_ns) ## in nanoseconds #1./40.) # 1/sampling rate in GHz
            DDFILT = np.array([(np.abs(FREQ)<dd_bwd) * np.cos(FREQ/dd_bwd*np.pi/2.) * (1.-np.cos(FREQ/dd_bwd*np.pi*2))]*data.shape[1]).T
            #LFILT = np.array([1./(1j*FREQ + logic_bwd)]*data.shape[1]).real.T
            fmin = np.abs(FREQ[1])
            LFILT = np.tile( np.array([np.tanh(abs(f)/logic_bwd)/(1j*f/logic_bwd + 1) for f in FREQ]).real,(data.shape[1],1)).T
            histinds = np.where((times>(t0+1))*(times<450))
            b=np.linspace(1.0,6,2**12+1)
            h0 = np.histogram(logtimes[histinds],bins=b)[0] 

        if batch ==0:
            sumsig = np.sum(data,axis=1)
            np.savetxt('%s/%s.samplesig'%(path,fname),np.column_stack((times,data)))
        else:
            sumsig += np.sum(data,axis=1)
        
        np.savetxt('%s/%s.sumsig'%(path,fname),np.column_stack((times,sumsig)))
    

        DATA = np.fft.fft(data.copy(),axis=0)
        DD = DDFILT*DATA
        dderiv = np.fft.ifft(DD,axis=0).real
        logic = dderiv*data
        logic[np.where(logic<0)] = 0
        
        if batch%50==0:
            np.savetxt('%s/%s.%i.fft'%(path,fname,batch),np.column_stack( (FREQ,np.abs(DD)) ))
            np.savetxt('%s/%s.%i.back'%(path,fname,batch),np.column_stack( (times,(dderiv*data)) ))
            np.savetxt('%s/%s.%i.logic'%(path,fname,batch),np.column_stack( (times,logic) ))
        DENOM = np.fft.fft(logic,axis=0)
        NUM = np.fft.fft(logtimes*logic,axis=0)
        num = np.fft.ifft( NUM * LFILT,axis=0).real # Fourier windowed integral
        denom = np.fft.ifft( DENOM * LFILT,axis=0).real # Fourier windowed integral
        logtimesout = num/denom
        hmat = np.array([(np.histogram(logtimesout[histinds,i],bins=b)[0]*np.exp(-b[:-1])) for i in range(logtimesout.shape[1])]).T
        hmat[np.where(hmat<histthresh)]=0
        if batch%50==0:
            np.savetxt('%s/%s.%i.logtimes'%(path,fname,batch),np.column_stack( (times,logtimesout) ) )
            np.savetxt('%s/%s.%i.histlogtimes'%(path,fname,batch),np.column_stack( (b[:-1],hmat) ) )
            np.savetxt('%s/%s.logtimes'%(path,fname),np.column_stack( (times,logtimesout) ) )
        if batch == 0:
            histsum = np.sum(hmat,axis=1)
        else:
            histsum += np.sum(hmat,axis=1)
        np.savetxt('%s/%s.cumhistlogtimes_%.1f_%.1fhistthresh'%(path,fname,logic_bwd,histthresh),np.column_stack( (b[:-1],histsum) ) )
        
    return
    
if __name__ == "__main__":
    main()
