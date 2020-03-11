#!/usr/bin/python3

import numpy as np
import lecroyparser 
import sys

def getHeaderBytes(fname):
    data = lecroyparser.ScopeData(fname)
    nvals = len(data.y)
    f = open(fname,'br')
    buf = f.read()
    f.close()
    nseek = int(len(buf) - nvals)
    return (nseek,nvals)

def laodArrayBytes(fname,nseek):
    f = open(fname,'br')
    buf = f.read()
    f.close()
    return np.array([np.int8(v) for v in buf])


def main():
    if len(sys.argv[1])==1:
        print('syntax is: loadbinary.py <path> <C1> <nwaves>')
    path = 'data_fs'
    if (len(sys.argv)>1):
        path = sys.argv[1]
    ch = 'C1'
    if (len(sys.argv)>2):
        ch = sys.argv[2]
    nwaves = 10
    if (len(sys.argv)>3):
        nwaves = int(sys.argv[3])
    ch_trig = 'C4'
    wv = int(0)
    fname = '%s/%s--%s--%05i.trc'%(path,ch,ch_trig,wv)
    (nseek,nvals) = getHeaderBytes(fname)
    data = np.zeros(nvals,dtype=np.int8)
    for wv in range(nwaves):
        fname = '%s/%s--%s--%05i.trc'%(path,ch,ch_trig,wv)
        oname = '%s/%s--%s--%05i.out'%(path,ch,ch_trig,wv)
        np.savetxt(oname,laodArrayBytes(fname,nseek),fmt = '%i')
    return

if __name__ == "__main__":
    main()
