#!/usr/bin/python3

from typing import List
import numpy as np
import h5py
import sys
import os
import re
import math

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

'''
def getHeaderBytesTimes(fname):
    #seems 355 bytes in header
    data = lecroyparser.ScopeData(fname)
    nvals = len(data.y)
    f = open(fname,'br')
    f.seek(HEADLEN)
    buf = f.read()
    f.close()
    nseek = int(len(buf) - 2*nvals)
    return (nseek,nvals,data.x)
'''

def traces2list(fnames:List):
    data = []
    for fname in fnames: 
        with open(fname,'br') as f:
            data += [trace2array(fname)]
    return data

def trace2array(fname):
    HEADLEN = 357
    array = np.array([],dtype=np.int16)
    with open(fname,'br') as f:
        f.seek(HEADLEN)
        array = np.frombuffer(f.read(),dtype=np.int16)
    if array.shape[0] != 0:
        return array
    else:
        print('failed to load %s'%fname)
        return None
