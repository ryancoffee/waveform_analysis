#!/usr/bin/python3

import numpy as np
import lecroyparser 
import h5py
import sys
import os
import re
import math

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
        print('syntax is: ./src/load_andrei_powerspectrum.py <datafile>')
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
    ninstances = 256
    nbatches = 32 
    times = []
    logtimes = []

    tstep_ns = 1./40
    t0=35
    for batch in range(nbatches):
        print('Processing batch %i of %i instances'%(batch,ninstances))
        raw = np.fromfile('%s/%s'%(path,fname),count=sz*ninstances,offset=batch*ninstances*sizeoftrace,dtype=float)
        data = raw.reshape(ninstances,sz).T
        if batch == 0:
            times = np.array([tstep_ns * i for i in range(data.shape[0])])
            out = np.zeros(data.shape[0])

        d = np.power(np.sum(np.abs(np.fft.fft(data,axis=0)),axis=1),int(2))
        #d -= np.mean(d[sz//4:3*sz//8])
        #d *= (d>0)
        #d *= float(2.**254)/float(np.max(d))
        #d += 1.
        #d = np.log2(d.astype(float))
        #print(np.min(d),np.max(d))
        out += d

    indlim=sz//2-sz//16
    np.savetxt('%s/%s_powerspect.out'%(path,fname),np.column_stack((np.fft.fftfreq(d.shape[0],tstep_ns)[:indlim],d[:indlim])))
    return

if __name__ == '__main__':
    main()
    
