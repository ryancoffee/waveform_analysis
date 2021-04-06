#!/usr/bin/python3

import numpy as np
from scipy import fft

def gauss(t,t0,w):
    y = np.exp(-1*np.power((t-float(t0))/float(w),int(2)))
    return y

def main():
    t = np.arange(256)
    s = 10*gauss(t,10,15)*np.cos(t*2*np.pi/7) + 20*gauss(t,105,20)*np.cos(t*2*np.pi/11)
    #_ = [print(' '*(20+int(v)) + '|') for v in s]
    y = np.concatenate((s,np.flip(s))) 
    #_ = [print(' '*(20+int(v)) + '+') for v in y]
    Y=fft.fft(y)
    YC = fft.dct(y)
    sz=y.shape[0]
    np.savetxt('powers.out',np.column_stack((Y.real,YC)))
    #print(len(s),len(y),len(Y),len(YC))
    #_ = [print(' '*(20+int(v)//20) + '.') for v in YC]

    return

if __name__ == '__main__':
    main()

