#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Module: r4idxmaps.py
# Generates index maps for radix-4 input-output reversals
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
    print "Error: Bad length argumnet '{}'".format(N)
    sys.exit(0)
N = int(sys.argv[1])
if N not in [16, 64, 256, 1024, 4096]:
    print 'Error: {} is not a valid r4-FFT length'.format(N)
    sys.exit(0)

N = np.int32(N)
idxary = np.zeros(N, dtype=np.uint16)

if N == 16:
    for j in xrange(4):
        for i in xrange(4):
            idxary[i+4*j] = 4*i+j
if N == 64:
    for k in xrange(4):
        for j in xrange(4):
            for i in xrange(4):
                idxary[i+4*j+16*k] = 16*i+4*j+k
if N == 256:
    for l in xrange(4):
        for k in xrange(4):
            for j in xrange(4):
                for i in xrange(4):
                    idxary[i+4*j+16*k+64*l] = 64*i+16*j+4*k+l
if N == 1024:
    for m in xrange(4):
        for l in xrange(4):
            for k in xrange(4):
                for j in xrange(4):
                    for i in xrange(4):
                        idxary[i+4*j+16*k+64*l+256*m] = 256*i+64*j+16*k+4*l+m
if N == 4096:
    for n in xrange(4):
        for m in xrange(4):
            for l in xrange(4):
                for k in xrange(4):
                    for j in xrange(4):
                        for i in xrange(4):
                            idxary[i+4*j+16*k+64*l+256*m+1024*n] = \
                                    1024*i+256*j+64*k+16*l+4*m+n

npyfname = './idxmaps/n' + str(N) + 'idxmap.npy'
txtfname = './idxmaps/n' + str(N) + 'idxmap.idx'
np.save(npyfname, idxary)
np.savetxt(txtfname, idxary, fmt='%d')
print '{}-size index array saved in {} & {}'.format(N, npyfname, txtfname)
