#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Module: r4tfs.py
# Generates twiddle factors for radix-4 FFTs and stores them in two formats in
# a subdirectory named 'twiddle.'  The two formats are '.npy' and text. The
# former can be loaded into a the correct dtype array with load().
#
# Returns:
# w[(N/4)][4] array of WN[n] = exp[-j*(2*pi/N)*n]
# (Note: at the negligible expense of some extra multiplications by 1.0f, the
#  twiddle factors for 0th row and column (n=0) are included in the returned
#  array.)
#
# Rashad Barghouti (UNI:rb3074)
# ELEN E4750, Fall 2015
#------------------------------------------------------------------------------
import sys
import numpy as np

# Read command line
if len(sys.argv) == 1:
    print 'Error: missing argument'
    sys.exit(0)
N = sys.argv[1]
if N.isdigit() == False:
    print "Error: Bad FFTlength argument '{}'".format(N)
    sys.exit(0)
N = int(sys.argv[1])
if N not in [16, 64, 256, 1024, 4096]:
    print 'Error: {} is not a valid FFT length'.format(N)
    sys.exit(0)
N = np.int32(N)
L = N >> 1
n = -np.pi/L
L = L >> 1

# Create Lx4 output array
w = np.empty((L, 4), dtype=np.complex64)
for k in xrange(L):
    w[k] = [1.0 + 0.0*1j,
            np.cos(n*k) + np.sin(n*k)*1j,
            np.cos(2*n*k) + np.sin(2*n*k)*1j, 
            np.cos(3*n*k) + np.sin(3*n*k)*1j]
npyfname = './twiddle/w' + str(N) + '.npy'
txtfname = './twiddle/w' + str(N) + '.tfs'
np.save(npyfname, w)
np.savetxt(txtfname, w, fmt='%.7f')
print '{}x4 twiddle factors saved in {} & {}'.format(L, npyfname, txtfname)
