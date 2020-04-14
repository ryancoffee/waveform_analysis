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
    return W

def getacFilter(data,FREQ,cut = 0.2):
    AC = np.ones(data.shape[0],dtype=float)
    inds = np.where(np.abs(FREQ)<cut)
    AC[inds] = 0.5*(1. - np.cos(FREQ[inds]*np.pi/cut))
    return AC



def main():
    if len(sys.argv)<2:
        print('syntax is: ./src/loadbinary.py <path/fname_front--(not numnber, not extension)> <nwaves> <roll filter vals>')
        return
    path = 'data_fs'
    if (len(sys.argv)>1):
        path = sys.argv[1]
    nwaves = 10
    if (len(sys.argv)>2):
        nwaves = int(sys.argv[2])
    nroll = 300
    if (len(sys.argv)>3):
        nroll = int(sys.argv[3])

    wv = int(0)
    fname = '%s%05i.trc'%(path,wv)
    hname = '%s.hist'%(path)
    (nseek,nvals,times) = getHeaderBytesTimes(fname)
    
    parseddata = np.zeros(nvals,dtype=np.float32)
    data = np.zeros(nvals,dtype=np.int16)
    FREQ = np.fft.fftfreq(data.shape[0],(times[1]-times[0])*1.0e9) ## in nanoseconds #1./40.) # 1/sampling rate in GHz
    W = getWeinerFilter(data,FREQ,cut = 3.0,noise = 0.1)
    W_lowpass = getWeinerFilter(data,FREQ,cut = 1.0,noise = 0.0001)
    AC = getacFilter(data,FREQ,cut = 0.05)
    w_ac_filter = np.roll(np.fft.ifft(W*AC).real,nroll)
    d_w_ac_filter = np.roll(np.fft.ifft(1j*FREQ*W*AC).real,nroll)
    headstring = 'times\tw_ac\tderiv_w_ac'
    np.savetxt('%s.filters'%(path),np.column_stack((times,w_ac_filter,d_w_ac_filter)),header=headstring)
    thresh = 1000 
    if (len(sys.argv)>3):
        thresh = int(sys.argv[3])
    negation = 1 # set to 1 for positive going signals, -1 for negative going
    if (len(sys.argv)>4):
        negation = int(sys.argv[4])


    tofs = []
    shots = int(0)
    hout = np.zeros(2**12,int)
    bins = np.linspace(0,2**14,hout.shape[0]+1)
    for wv in range(nwaves):
        fname = '%s%05i.trc'%(path,wv)
        if not os.path.exists(fname):
            continue
        #parseddata = lecroyparser.ScopeData(fname)
        oname = '%s%05i.out'%(path,wv)
        data = loadArrayBytes_int16(fname,nseek,nvals,byteorder = 'little')
        y = np.fft.ifft(np.fft.fft(negation*data)*W*AC).real
        dy = np.fft.ifft(1j*FREQ*np.fft.fft(negation*data)*W*AC).real
        tofs = tofs + zeroCrossings((y * dy)/float(y.shape[0]),thresh)
        shots += 1
        if wv%1000 == 0:
            headstring = '(data,y,dy,y*dy/float(y.shape[0]))'
            np.savetxt(oname,np.column_stack((times,data,y,dy,y*dy/float(y.shape[0]))),fmt= '%f',header=headstring)
            hout += np.histogram(tofs,bins)[0]
            headstring = 'shots,thresh,negation = (%i,%i,%i)'%(shots,thresh,negation)
            np.savetxt(hname,hout,fmt='%i',header = headstring)
            tofs = []
    hout += np.histogram(tofs,bins)[0]
    headstring = 'shots,thresh,negation = (%i,%i,%i)'%(shots,thresh,negation)
    np.savetxt(hname,hout,fmt='%i',header = headstring)
    return

if __name__ == "__main__":
    main()
