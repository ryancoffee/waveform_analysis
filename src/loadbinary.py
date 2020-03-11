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

def loadArrayBytes(fname,nseek):
    f = open(fname,'br')
    f.seek(nseek)
    buf = f.read()
    f.close()
    return np.array([np.int8(v) for v in buf],dtype=np.int8)


def main():
    if len(sys.argv[1])==1:
        print('syntax is: ./src/loadbinary.py <path> <C1 or so> <nwaves>')
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
    #parseddata = np.zeros(nvals,dtype=np.float32)
    data = np.zeros(nvals,dtype=np.int8)
    thresh = 6
    if (len(sys.argv)>4):
        thresh = int(sys.argv[4])

    for wv in range(nwaves):
        fname = '%s/%s--%s--%05i.trc'%(path,ch,ch_trig,wv)
        #parseddata = lecroyparser.ScopeData(fname)
        oname = '%s/%s--%s--%05i.out'%(path,ch,ch_trig,wv)
        data = loadArrayBytes(fname,nseek)
        if (wv%10==0) or (np.max(data)>np.mean(data)+thresh):
            np.savetxt(oname,data,fmt = '%i')
        #np.savetxt(oname + 'parser',parseddata.y,fmt = '%.4f')
    return

if __name__ == "__main__":
    main()
