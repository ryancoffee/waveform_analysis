#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import h5py
import sys

def main():
    x = np.arange(-10,10,dtype=float)
    X = tf.constant(np.column_stack((np.ones(len(x)),x,x*x,x*x*x,x*x*x,x*x*x*x)))
    y = tf.constant((1 + 3*x + .1*x**2 - .05*x**3 + .003*x**4 + np.random.normal(0,10,len(x))).reshape((-1,1)))
    print(X)
    print(y)

    Xinv = tf.linalg.pinv(X,transpose_a=True)
    theta = tf.linalg.matvec(Xinv,y)
    print(theta)
    return

if __name__ == '__main__':
    main()
