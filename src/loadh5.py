#!/usr/bin/python3

import numpy as np
import lecroyparser 
import sys
import h5py

# for .wfm files from the Agilent, look into the following (https://pythonhosted.org/bitstring/introduction.html)

def getWeinerFilter(fname,cut = 0.075,noise =100.):
    f=h5py.File(fname,'r')
    data = np.array(f['Waveforms']['Channel 2']['Channel 2Data'][()], dtype = float)
    W = np.zeros(data.shape[0],dtype=float)
    DATA = np.fft.fft(data)/np.sqrt(data.shape[0])
    FREQ = np.fft.fftfreq(data.shape[0])
    inds = np.where(np.abs(FREQ)<cut)
    c2 = 0.5*(1.+np.cos(FREQ[inds]*np.pi/cut))
    W[inds] = c2 / (c2 + noise)
    return (W,FREQ)

def getacFilter(fname,cut = 0.075):
    f=h5py.File(fname,'r')
    data = np.array(f['Waveforms']['Channel 2']['Channel 2Data'][()], dtype = float)
    AC = np.ones(data.shape[0],dtype=float)
    FREQ = np.fft.fftfreq(data.shape[0])
    inds = np.where(np.abs(FREQ)<cut)
    AC[inds] = 0.5*(1. - np.cos(FREQ[inds]*np.pi/cut))
    return (AC,FREQ)

def applyWeinerFilter_acFilter(fname,W,AC):
    f=h5py.File(fname,'r')
    y = np.array(f['Waveforms']['Channel 2']['Channel 2Data'][()], dtype = float)
    Y = np.fft.fft(y)
    y_filt = np.fft.ifft(Y*W).real
    y_ac_filt = np.fft.ifft(Y*W*AC).real
    return (y, y_filt, y_ac_filt)

def applyWeinerFilter(fname,W):
    f=h5py.File(fname,'r')
    y = np.array(f['Waveforms']['Channel 2']['Channel 2Data'][()], dtype = float)
    Y = np.fft.fft(y)
    y_filt = np.fft.ifft(Y*W).real
    return (y, y_filt)

def getHeaderBytes_h5(fname):
    f=h5py.File(fname,'r')
    nvals = len(f['Waveforms']['Channel 2']['Channel 2Data'][()])
    f.close()
    f = open(fname,'br')
    buf = f.read()
    f.close()
    # careful, buf is now a list of 16 bit integers, so each val takes 2 bytes.
    nseek = int(len(buf) - 2*nvals)
    return (nseek,nvals)

def getHeaderBytes(fname):
    data = lecroyparser.ScopeData(fname)
    nvals = len(data.y)
    f = open(fname,'br')
    buf = f.read()
    f.close()
    nseek = int(len(buf) - nvals)
    return (nseek,nvals)

def loadArrayBytes_h5(fname,nseek):
    data =[]
    # careful, buf is now a list of 16 bit integers, so each val takes 2 bytes.
    return np.array(data,dtype=np.int16)

def loadArrayBytes(fname,nseek):
    f = open(fname,'br')
    f.seek(nseek)
    buf = f.read()
    f.close()
    return np.array([np.int8(v) for v in buf],dtype=np.int8)


def main():
    if len(sys.argv[1])==1:
        print('syntax is: ./src/loadh5.py <path+filehead> <nwaves> <cutoff>')
    path = 'data_fs'
    if (len(sys.argv)>1):
        path = sys.argv[1]
    nwaves = 10
    if (len(sys.argv)>2):
        nwaves = int(sys.argv[2])
    wv = int(0)
    fname = '%s%05i.h5'%(path,wv)
    c = 0.05
    n = 100.
    if (len(sys.argv)>3):
        c = int(sys.argv[3])
    if (len(sys.argv)>4):
        n = int(sys.argv[4])
    (WFILTER,FREQ) = getWeinerFilter(fname,cut = c,noise = n)
    (ACFILTER,FREQ) = getacFilter(fname,cut = .025*c)

    for wv in range(nwaves):
        fname = '%s%05i.h5'%(path,wv)
        oname = '%s.out'%(fname)
        np.savetxt(oname,np.column_stack( applyWeinerFilter_acFilter(fname,WFILTER,ACFILTER) ),fmt = '%f')
    return

if __name__ == "__main__":
    main()
