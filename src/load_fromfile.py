#!/usr/bin/python3

import numpy as np
import lecroyparser 
import h5py
import sys
import os
import re
import math

def zeroCrossings2energy(data,thresh,tstep = 1.,t0 = 50.):
    # from fitting Ave simulations with theta0 = memmap([ 6.39445393, -0.03775023, -0.46237318])
    # this is log(t - t0[ns]) = [6.39445393, -0.03775023, -0.46237318].T dot [1,ln(vret[V]),ln(ekin[eV])]
    #                                                       
    # fit f(x) energiesfile u (log($1-50)):(log(1.55*($0+1)+30)) via a,b,c,d
    # f(x)= a + b*x + c* x**2 + d* x**3
    # Final set of parameters            Asymptotic Standard Error
    # =======================            ==========================
    # a               = -99.8063         +/- 10.31        (10.33%)
    # b               = 72.8784          +/- 6.79         (9.317%)
    # c               = -16.7728         +/- 1.491        (8.886%)
    # d               = 1.26497          +/- 0.109        (8.616%)
    # new fitting using also the 20Vacc results
    # Final set of parameters            Asymptotic Standard Error
    # =======================            ==========================
    # a               = 13.0505          +/- 0.259        (1.985%)
    # b               = -2.96445         +/- 0.1147       (3.869%)
    # c               = 0.199342         +/- 0.01267      (6.354%)
    # 
    # better points from data_fs/energies.notes file (based on 200 threshold for 20Vacc case, still using 1000 threshold for 30Vacc case
    # Final set of parameters            Asymptotic Standard Error
    # =======================            ==========================
    # a               = 12.6565          +/- 0.2341       (1.85%)
    # b               = -2.79381         +/- 0.1044       (3.737%)
    # c               = 0.18091          +/- 0.01161      (6.417%)


    c = np.array([12.6565,-2.79381,0.18091])
    tofs = zeroCrossings(data,thresh,tstep)
    logt = [math.log(v-t0) for v in tofs if (v>t0)]
    X_f = np.row_stack((np.ones(len(logt)),logt,np.power(logt,2)))
    e = list(np.exp(c.dot(X_f)))
    return tofs,e

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

def checktracelength(raw):
    ntest = raw.shape[0]//20
    Y = np.abs(np.fft.fft(raw[:ntest]))
    F = np.fft.fftfreq(ntest)
    i = np.argmax(Y[10:len(Y)//4])
    n = int((1./F[i] + 500)/1e3)*1000 + 2
    instances = raw.shape[0]//n
    print('how do you like as the sample intervals:\t%i'%n)
    return (n,instances)

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
        print('syntax is: ./src/load_fromfile.py <datafile>')
        return
    m = re.match('(.+)/(.+)$',sys.argv[1])
    if m:
        path = m.group(1)
        fname = m.group(2)
    else:
        print('Failed the filename match')
        return

    sizeoftrace = getAndreiSize('%s/%s_HEADER.txt'%(path,fname))
    sz = sizeoftrace//8
    ninstances = 100
    nbatches = 100
    for batch in range(nbatches):
        data = np.fromfile('%s/%s'%(path,fname),count=sz*ninstances,offset=batch*ninstances*sizeoftrace,dtype=float).reshape(ninstances,sz).T
        tstep_ns = 1./40
        times = np.array([tstep_ns * i for i in range(data.shape[0])])

        if batch ==0:
            sumsig = np.sum(data,axis=1)
        else:
            sumsig += np.sum(data,axis=1)
        
        np.savetxt('%s/%s.sumsig'%(path,fname),np.column_stack((times,sumsig)))
        np.savetxt('%s/%s.sig'%(path,fname),np.column_stack((times,data)))
    
        print('using as tstep = \t%f'%(tstep_ns))

        DATA = np.fft.fft(data.copy(),axis=0)
        FREQ = np.fft.fftfreq(data.shape[0],tstep_ns) ## in nanoseconds #1./40.) # 1/sampling rate in GHz
        bwd = 1.5 # this is in GHz
        DDFILT = (np.abs(FREQ)<bwd) * np.cos(FREQ/bwd*np.pi/2.) * (1.-np.cos(FREQ/bwd*np.pi*2))
        DD = np.array([DDFILT*DATA[:,i] for i in range(DATA.shape[1])])
        out = np.fft.ifft(DD,axis=1).real
        np.savetxt('%s/%s.fft'%(path,fname),np.column_stack( (FREQ,np.abs(DD.T)) ))
        np.savetxt('%s/%s.back'%(path,fname),np.column_stack( (times,(out.T*data)) ))
        logtimes = np.log(times-50 + 1j*tstep_ns).real
        #logtimes = times
        #NUM = np.row_stack([ np.fft.fft(out[i,:]*data[:,i]*logtimes) * .5*(1.+ np.cos(FREQ/bwd*np.pi) * (np.abs(FREQ)<bwd)) for i in range(data.shape[1])])
        #DENOM = np.row_stack([np.fft.fft(out[i,:]*data[:,i]) * .5*(1.+ np.cos(FREQ/bwd*np.pi) * (np.abs(FREQ)<bwd)) for i in range(data.shape[1])]) 
        #NUM = np.column_stack([ np.fft.fft(logtimes) * .5*(1.+ np.cos(FREQ/bwd*np.pi) * (np.abs(FREQ)<bwd)) for i in range(data.shape[1])])
        DENOM = np.column_stack([ np.fft.fft(out[i,:]*data[:,i]) for i in range(data.shape[1])])
        NUM = np.column_stack([ np.fft.fft(logtimes*out[i,:]*data[:,i]) for i in range(data.shape[1])])
        print(NUM.shape)
        print(FREQ.shape)
        bwd=.15
        num = [np.fft.ifft( NUM[:,i] * np.exp(-(FREQ/bwd)**2) / (1j*FREQ + 1)).real for i in range(NUM.shape[1])]
        denom = [np.fft.ifft( DENOM[:,i] * np.exp(-(FREQ/bwd)**2) / (1j*FREQ + 1)).real for i in range(DENOM.shape[1])]
        #DENOM = np.row_stack([np.fft.fft(out[i,:]*data[:,i]) * .5*(1.+ np.cos(FREQ/bwd*np.pi) * (np.abs(FREQ)<bwd)) for i in range(data.shape[1])]) 
        #denom = np.fft.ifft(DENOM,axis=1).real
        logtimesout = np.array(num).T/np.array(denom).T
        print(logtimesout.shape)
        print('here we want the expectation of <log(t)>')
        np.savetxt('%s/%s.logtimes'%(path,fname),np.column_stack( (times,logtimesout) ) )
    
        b=np.linspace(1.0,6,2**12+1)
        #b=np.linspace(50,500,2**10+1)
        inds = np.where((times>(50+1))*(times<450))
        h0 = np.histogram(logtimes[inds],bins=b)[0] 
        hmat = np.array([sigmoid((np.histogram(logtimesout[inds,i],bins=b)[0]-h0)*np.exp(-b[:-1]),.25,.025) for i in range(logtimesout.shape[1])]).T
        #hmat = np.array([(np.histogram(logtimesout[:,i],bins=b)[0]) for i in range(logtimesout.shape[1])]).T
    
        np.savetxt('%s/%s.histlogtimes'%(path,fname),np.column_stack( (b[:-1],hmat) ) )
    
        if batch ==0:
            hout = np.sum(hmat,axis=1)
        else:
            hout += np.sum(hmat,axis=1)
        np.savetxt('%s/%s.cumhistlogtimes'%(path,fname),np.column_stack( (b[:-1],hout) ) )
    
    
        
    
    return
    
    
    
    
'''
    W = getWeinerFilter(data,FREQ,cut = 3.0,noise = 0.1)
    W_lowpass = getWeinerFilter(data,FREQ,cut = 1.0,noise = 0.0001)
    AC = getacFilter(data,FREQ,cut = 0.05)
    w_ac_filter = np.roll(np.fft.ifft(W*AC).real,nroll)
    d_w_ac_filter = np.roll(np.fft.ifft(1j*FREQ*W*AC).real,nroll)
    headstring = 'times\tw_ac\tderiv_w_ac'
    np.savetxt('%s.filters'%(path),np.column_stack((times,w_ac_filter,d_w_ac_filter)),header=headstring)
    thresh = 200 
    if (len(sys.argv)>3):
        thresh = int(sys.argv[3])
    negation = 1 # set to 1 for positive going signals, -1 for negative going
    if (len(sys.argv)>4):
        negation = int(sys.argv[4])


    tofs = []
    ens = []
    shots = int(0)
    hout = np.zeros(2**12,int)
    eout = np.zeros(2**12,int)
    tbins = np.linspace(0,2**9,hout.shape[0]+1)
    ebins = np.linspace(0,256,eout.shape[0]+1)
    for wv in range(nwaves):
        fname = '%s/%s--%05i.trc'%(path,fname_front,wv)
        if not os.path.exists(fname):
            continue
        #parseddata = lecroyparser.ScopeData(fname)
        oname = '%s/%s--%05i.out'%(path,fname_front,wv)
        data = loadArrayBytes_int16(fname,nseek,nvals,byteorder = 'little')
        y = np.fft.ifft(np.fft.fft(negation*data)*W*AC).real
        dy = np.fft.ifft(1j*FREQ*np.fft.fft(negation*data)*W*AC).real
        t,e = zeroCrossings2energy((y * dy)/float(y.shape[0]),thresh,tstep = tstep_ns,t0=50.)
        tofs = tofs + list(t)
        ens = ens + list(e)
        shots += 1
        if wv%1000 == 0:
            headstring = '(data,y,dy,y*dy/float(y.shape[0]))'
            np.savetxt(oname,np.column_stack((times,data,y,dy,y*dy/float(y.shape[0]))),fmt= '%f',header=headstring)
            hout += np.histogram(tofs,tbins)[0]
            eout += np.histogram(ens,ebins)[0]
            headstring = 'shots,thresh,negation = (%i,%i,%i)'%(shots,thresh,negation)
            np.savetxt(hname,np.column_stack((tbins[:-1],hout)),fmt='%.2f',header = headstring)
            np.savetxt(ename,np.column_stack((ebins[:-1],eout)),fmt='%.2f',header = headstring)
            tofs = []
            ens = []
    hout += np.histogram(tofs,tbins)[0]
    eout += np.histogram(ens,ebins)[0]
    headstring = 'shots,thresh,negation = (%i,%i,%i)'%(shots,thresh,negation)
    np.savetxt(hname,np.column_stack((tbins[:-1],hout)),fmt='%.2f',header = headstring)
    np.savetxt(ename,np.column_stack((ebins[:-1],eout)),fmt='%.2f',header = headstring)
    return
    '''

if __name__ == "__main__":
    main()
