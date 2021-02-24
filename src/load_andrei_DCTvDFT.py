#!/usr/bin/python3

import numpy as np
from scipy import fft
import re
import sys
import time
from threading import get_ident

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

    path = ''
    fname = ''
    if len(sys.argv)<2:
        print('syntax is: ./src/load_andrei_DCTvDFT.py <datafile>')
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
    ninstances = 32
    nbatches = 4 
    tstep_ns = 1./40

    tid = get_ident()
    clkID=time.pthread_getcpuclockid(tid)

    timeslist = []
    dft_timeslist = []
    for batch in range(nbatches):
        print('Processing batch %i of %i instances'%(batch,ninstances))
        raw = np.fromfile('%s/%s'%(path,fname),count=sz*ninstances,offset=batch*ninstances*sizeoftrace,dtype=float)
        data = raw.reshape(ninstances,sz).T
        slim = 2**int(np.log2(data.shape[0]))
        lim = int(2**12)
        print(slim,lim)
        d = np.row_stack((data[:slim,:],np.flipud(data[:slim,:])))
        freqs = np.fft.fftfreq(d.shape[0])
        freqsmat = np.tile(freqs,(d.shape[1],1)).T
        frq = np.arange(lim,dtype=float)
        flt1d = frq*(1.+np.cos(frq*np.pi/frq.shape[0]))
        filt = np.tile(flt1d,(d.shape[1],1)).T
        t_0 = time.clock_gettime_ns(clkID)
        DC = fft.dct(d,axis=0)
        #DC[lim:2*lim,:] = 0
        DC[lim:,:] = 0
        DC[:lim,:] *= filt 
        #DS[5000:,:] = 0
        dcc = fft.idct(DC[:2*lim,:],axis=0)
        dsc = fft.idst(DC[:2*lim,:],axis=0)
        #dcc = fft.idct(DC,axis=0)
        #dsc = fft.idst(DC,axis=0)
        timeslist += [time.clock_gettime_ns(clkID) - t_0]
        logic = dsc*dcc
        np.savetxt('%s/%s_b%i_sample.dat'%(path,fname,batch),d*1e5,fmt='%i')
        np.savetxt('%s/%s_b%i_dct.dat'%(path,fname,batch),DC,fmt='%.3f')
        #np.savetxt('%s/%s_b%i_dst.dat'%(path,fname,batch),DS,fmt='%.3f')
        np.savetxt('%s/%s_b%i_idct.dat'%(path,fname,batch),dcc,fmt='%.3f')
        np.savetxt('%s/%s_b%i_idst.dat'%(path,fname,batch),dsc,fmt='%.3f')
        np.savetxt('%s/%s_b%i_logic.dat'%(path,fname,batch),logic[:lim,:],fmt='%.3f')
        t_0 = time.clock_gettime_ns(clkID)
        D = np.fft.fft(d,axis=0)
        DD = 1j*freqsmat*D
        logic = np.fft.ifft(D*DD,axis=0).real
        dft_timeslist += [time.clock_gettime_ns(clkID) - t_0]
        np.savetxt('%s/%s_b%i_dft_logic.dat'%(path,fname,batch),logic,fmt='%.3f')
        

    print(timeslist)
    print(dft_timeslist)

    return

if __name__ == '__main__':
    main()
