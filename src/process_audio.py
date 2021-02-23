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
            if i == sz-1: break
        if i == sz-1: break
        while data[i] > 0:
            i += 1
            if i == sz-1: break
        tofs = tofs + [1./float(data[i]-data[i-1])*data[i] + float(i)]
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
    return np.tile(W,data.shape[1])

def getacFilter(data,FREQ,cut = 0.2):
    AC = np.ones(data.shape[0],dtype=float)
    inds = np.where(np.abs(FREQ)<cut)
    AC[inds] = 0.5*(1. - np.cos(FREQ[inds]*np.pi/cut))
    return np.tile(AC,data.shape[1])



def main():
    if len(sys.argv[1])==1:
        print('syntax is: ./src/loadbinary.py <path/fname_front> <nstart> <nwaves> <negation(1,-1)> <nrolls> <overlap>')
    path = 'data_fs'
    if (len(sys.argv)>1):
        path = sys.argv[1]
    nstart = 0
    nwaves = 10
    if (len(sys.argv)>2):
        nstart = int(sys.argv[2])
    if (len(sys.argv)>3):
        nwaves = int(sys.argv[3])
    wv = int(0)
    fname = '%s%05i.trc'%(path,wv)
    hname = '%s.hist'%(path)
    (nseek,nvals,times) = getHeaderBytesTimes(fname)
    print(nvals,times[2]-times[1])

    negation = 1 # set to 1 for positive going signals, -1 for negative going
    if (len(sys.argv)>4):
        negation = int(sys.argv[4])
    
    nrolls=20
    overlap=2
    if (len(sys.argv)>5):
        nrolls = int(sys.argv[5])
    if (len(sys.argv)>6):
        overlap = int(sys.argv[6])

    parseddata = np.zeros(nvals,dtype=np.float32)
    data = np.zeros((nvals//nrolls,nrolls*overlap),dtype=np.int16)
    sz = data.shape[0]
    window = 0.5*(1.+np.sin(np.arange(sz)*np.pi/sz))
    outdata = np.zeros(nvals//nrolls,dtype=np.float32)
    dt = times[1]-times[0]
    FREQ = np.fft.fftfreq(data.shape[0],dt) 
    print(data.shape)
    print(FREQ[:5])
    print('Frequency step in Hz: %.3f'%(1./data.shape[0]/dt))
    W = getWeinerFilter(data,FREQ,cut = 4e3,noise = 0.01)
    W_lowpass = getWeinerFilter(data,FREQ,cut = 1.0,noise = 0.0001)
    AC = getacFilter(data,FREQ,cut = 0.05)


    for wv in range(nstart,nstart+nwaves):
        fname = '%s%05i.trc'%(path,wv)
        if not os.path.exists(fname):
            continue
        fulldata = loadArrayBytes_int16(fname,nseek,nvals,byteorder = 'little')
        for i in range(nrolls*overlap):
            fulldata = np.roll(fulldata,-sz//overlap)
            data[:,i] = fulldata[:sz]*window
        Y = np.abs(np.fft.fft(negation*data,axis=0)).real
        outdata = np.column_stack((outdata,Y))
        if wv%1 == 0:
            oname = '%s%05i.powerspec'%(path,wv)
            headstring = 'powerspec'
            np.savetxt(oname,Y,fmt= '%f',header=headstring)
            oname = '%s.powerspec'%(path)
            headstring = 'rolling powerspec'
            np.savetxt(oname,outdata,fmt= '%f',header=headstring)

    return

if __name__ == "__main__":
    main()
