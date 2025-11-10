#!/usr/bin/python3

from typing import List
import numpy as np
import h5py
import sys
import os
import re
import math

def txts2list(fnames:List):
    data = []
    for fname in fnames:
        data += [ txt2array(os.path.join(fname)) ]
    return data

def txt2array(fname):
    array = np.loadtxt(fname,skiprows=5,delimiter=',',usecols=(1),dtype=np.float32) * 1e4
    #units are 0.1mV
    return array.astype(np.int16)

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
