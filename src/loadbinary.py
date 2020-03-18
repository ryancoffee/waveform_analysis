#!/usr/bin/python3

import numpy as np
import lecroyparser 
import h5py
import sys
import os

def sigmoid(x,c,w):
    return 1./(1.+np.exp(-(x-c)/w))

def zeroCrossings(data,thresh):
    tofs = []
    i = int(0)
    sz = data.shape[0]
    while i < data.shape[0]-1:
        while data[i] < thresh:
            i += 1
            if i == sz: break
        if i == sz: break
        while data[i] > 0:
            i += 1
            if i == sz: break
        tofs = tofs + [1./float(data[i]-data[i-1])*data[i] + float(i)]
        if i == sz: break
        while data[i] < 0:
            i += 1
            if i == sz: break
    return tofs 

def getHeaderBytes(fname):
    data = lecroyparser.ScopeData(fname)
    nvals = len(data.y)
    f = open(fname,'br')
    buf = f.read()
    f.close()
    nseek = int(len(buf) - 2*nvals)
    return (nseek,nvals)

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
    DATA = np.fft.fft(data)
    inds = np.where(np.abs(FREQ)<cut)
    c2 = 0.5*(1.+np.cos(FREQ[inds]*np.pi/cut))
    W[inds] = c2 / (c2 + noise)
    return W

def getacFilter(data,FREQ,cut = 0.2):
    AC = np.ones(data.shape[0],dtype=float)
    inds = np.where(np.abs(FREQ)<cut)
    AC[inds] = 0.5*(1. - np.cos(FREQ[inds]*np.pi/cut))
    return AC



def main():
    if len(sys.argv[1])==1:
        print('syntax is: ./src/loadbinary.py <path> <fname_front, not extension> <nwaves>')
    path = 'data_fs'
    if (len(sys.argv)>1):
        path = sys.argv[1]
    fname_front = 'C1--ATI_attempt_03_16_2020--'
    if (len(sys.argv)>2):
        fname_front = sys.argv[2]
    nwaves = 10
    if (len(sys.argv)>3):
        nwaves = int(sys.argv[3])
    wv = int(0)
    fname = '%s/%s%05i.trc'%(path,fname_front,wv)
    hname = '%s/%s.hist'%(path,fname_front)
    (nseek,nvals) = getHeaderBytes(fname)
    
    parseddata = np.zeros(nvals,dtype=np.float32)
    data = np.zeros(nvals,dtype=np.int16)
    FREQ = np.fft.fftfreq(data.shape[0],1./40.) # 1/sampling rate in GHz
    W = getWeinerFilter(data,FREQ,cut = 3.0,noise = 0.1)
    W_lowpass = getWeinerFilter(data,FREQ,cut = 1.0,noise = 0.0001)
    AC = getacFilter(data,FREQ,cut = 0.05)
    thresh = 6
    negation = 1 # set to 0 for positive going signals, 1 or -1 for negative going
    if (len(sys.argv)>4):
        thresh = int(sys.argv[4])

    tofs = []
    hout = np.zeros(2**12,int)
    bins = np.linspace(0,2**14,hout.shape[0]+1)
    for wv in range(nwaves):
        fname = '%s/%s%05i.trc'%(path,fname_front,wv)
        if not os.path.exists(fname):
            continue
        #parseddata = lecroyparser.ScopeData(fname)
        oname = '%s/%s%05i.out'%(path,fname_front,wv)
        data = loadArrayBytes_int16(fname,nseek,nvals,byteorder = 'little')
        y = np.fft.ifft(np.fft.fft(data)*W*AC).real
        dy = np.fft.ifft(1j*FREQ*np.fft.fft(data)*W*AC).real
        #DATA = np.fft.fft(sigmoid(y,2500,1000))*W_lowpass
        #IDATA = np.fft.fft(sigmoid(y,2500,1000))*np.arange(y.shape[0])*W_lowpass
        #num = np.fft.ifft(IDATA).real
        #denom = np.fft.ifft(DATA).real
        tofs = tofs + zeroCrossings((y * dy)/float(y.shape[0]),1000)
        if wv%1000 == 0:
            np.savetxt(oname,np.column_stack((data,y,dy)),fmt= '%f')
            hout += np.histogram(tofs,bins)[0]
            np.savetxt(hname,hout,fmt='%i')
            tofs = []
    hout += np.histogram(tofs,bins)[0]
    np.savetxt(hname,hout,fmt='%i')
    return

if __name__ == "__main__":
    main()
