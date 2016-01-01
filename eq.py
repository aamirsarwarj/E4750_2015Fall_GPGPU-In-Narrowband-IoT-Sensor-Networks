#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Module: nbeq.py
#
# Rashad Barghouti (UNI:rb3074)
# AM Sarwar Jahan (UNI:aj2599)
# ELEN E4750, Fall 2015
#------------------------------------------------------------------------------

FRAMES_BATCH_SIZE = 32
TX_FRAME_LEN = 500
SNRdB = 20
QAM_SIZE = 4
# If FFT size is defined here, it will be used to determine frame sizes.
# Otherwise, all will be derived from TX_FRAME_LENGTH
FFT_SIZE = None

# FFT kernel config dictionary
GLOBAL_DIM0 = 0
LOCAL_DIM0 = 1
FFTSIZE_DEFINE = 2
OTHER_DEFINES = 3
kernelplan = {16: [1, 1, '#define FFTSIZE 16U', ''],
              64: [4, 4, '#define FFTSIZE 64U', ''],
              256: [16, 16, '#define FFTSIZE 256U', ''],
              1024: [16, 16, '#define FFTSIZE 1024U', ''],
              4096: [16, 16, '#define FFTSIZE 4096U', '']}

PLOT_MAGNITUDE_SPECTRA = False
PLOT_POWER_SPECTRA = False
SHOW_PLOTS = False
PRINT_KERNELS = False

# System imports
import os
import time
import pyopencl as cl
import pyopencl.array
import numpy as np
from scipy import signal
# To avoid "backend has already been chosen error," set GPU platform before
# import pyplot
if 'rb3074' in os.environ['HOME']:
    GPU = 'TESSERACT_K40'
    import matplotlib as mpl
    mpl.use('agg')
else:
    GPU = 'HOME_QUADRO'
import matplotlib.pyplot as plt


def _main():

###############################################################################
######################### Transmitter and Channel #############################
###############################################################################

    constellQAM4  = np.array([-1-1j, -1+1j,  1-1j,  1+1j], dtype=np.complex64)
    constellQAM16 = np.array([-3-3j, -3-1j, -3+3j, -3+1j,
                              -1-3j, -1-1j, -1+3j, -1+1j,
                               3-3j,  3-1j,  3+3j,  3+1j,
                               1-3j,  1-1j,  1+3j,  1+1j], dtype=np.complex64)

    # Define the channel as a 3-tap notch filter that produces a deep fade at
    # center frequencies and an attenuated 2nd path
    channel = np.array([1, 0, 0.5]).astype(np.float32)

    # N = Number of frames to generate. 
    # txfrmlen = Frame length in symbols. Default is 500 symbols, which is
    # equivalent to QAM4-modulated frames being transmitted every 1 ms on a
    # 1 Mbps link rate
    #
    N = FRAMES_BATCH_SIZE
    txfrmlen = TX_FRAME_LEN
    constell = constellQAM4 if QAM_SIZE == 4 else constellQAM16
    snrdB = SNRdB

    if FFT_SIZE == None: 
        fftlen = 2**np.int32(np.log2(txfrmlen*2-1)+1)
    else:
        fftlen = FFT_SIZE
        txfrmlen = (FFT_SIZE >> 1) - 1

    txpkts = np.empty((N, txfrmlen), dtype=np.int8)
    rxfrms = np.empty((N, fftlen), dtype=np.complex64)
    for n in xrange(N):

        txpkt, txfrm = QAMmodulate(txfrmlen, constell)

        # Fitler Tx frame with channel FIR and compute output signal avg.
        # symbol energy, Es.
        # Note: returned data type from lfilter() is double (complex128), so
        # need to downcast to complex64. Otherwise GPU ops will be on smaller
        # array size than desired
        chfrm = signal.lfilter(channel, [1, 0, 0], txfrm).astype(np.complex64)
        chEs = np.mean(np.abs(chfrm)**2)
        chEsdBm = 10*np.log10(chEs)

        # Add AWGN to produce the received frame; N0 is the equivalent lowpass
        # noise PSD
        rxfrm, N0 = awgn(chfrm, snrdB, chEs)
        rxEs = np.mean(np.abs(rxfrm)**2)
        rxEsdBm = 10.0*np.log10(rxEs)

        #np.set_printoptions(precision=2)
        #print 'Avg symbol energy before AWGN: chEs: {:.3f} dBm'.format(chEsdBm)
        #print 'Avg. received symbol energy: rxEs: {:.3f} dBm'.format(rxEsdBm)
        #print 'One-sided noise PSD (N0): {:.3f}'.format(N0)
        
        # Zero-pad rxfrm to fftlen (necessary for GPU processing)
        rxfrm.resize(fftlen, refcheck=False)

        rxfrms[n] = rxfrm
        txpkts[n] = txpkt 

###############################################################################
############################# Receiver Processing #############################
###############################################################################

    # Compute channel and equalizers' FFTs and plot their magnitude spectra
    chfft = np.fft.fft(channel, n=fftlen).astype(np.complex64)
    ZFfft = np.reciprocal(chfft)
    MMSEfft = np.reciprocal(chfft+N0)

    # Run Numpy implementation on N frames
    npyffttm = 0
    npyEqtm = 0
    npyiffttm = 0
    npyfrmRxtm = 0
    npyNfrmsRxtm = time.clock()
    for i in xrange(N):

        # Transform ith Rx frame
        rxfrm = rxfrms[i]
        npyfftstarttm = time.clock()
        rxfrmfft = np.fft.fft(rxfrm, fftlen)
        npyfftendtm = time.clock()

        # Do point-by-point multiplication with Equalizer ffts
        MMSEoutfft = rxfrmfft * MMSEfft
        npyEqstarttm = time.clock()
        ZFoutfft = rxfrmfft * ZFfft
        npyifftstarttm = time.clock()

        # Do inverse ffts to complete equalization/demodulation
        MMSEout = np.fft.ifft(MMSEoutfft)
        npyifftendtm = time.clock()
        ZFout = np.fft.ifft(ZFoutfft)
        npyifftendtm = time.clock()

        npyffttm += 1e3*(npyfftendtm-npyfftstarttm)
        npyEqtm += 1e3*(npyifftstarttm-npyEqstarttm)
        npyiffttm += 1e3*(npyifftendtm-npyifftstarttm)
        npyfrmRxtm += (npyffttm + npyEqtm + npyiffttm)

    npyNfrmsRxtm = 1e3*(time.clock()-npyNfrmsRxtm)
    npyavgffttm = npyffttm/N
    npyavgiffttm = npyiffttm/N
    npyavgfrmRxtm = npyfrmRxtm/N

    # Demodulate received symbols for last frame to recover user data
    MMSEdata, MMSEdatabinary = QAMdemodulate(MMSEout[:txfrmlen], constell)
    ZFdata, ZFdatabinary = QAMdemodulate(ZFout[:txfrmlen], constell)

    # Calculate bit error rates (BER).
    txbits = dec2bin(txpkts[N-1], M=constell.size)
    MMSEbiterrors = np.sum(np.abs(txbits-MMSEdatabinary))
    ZFbiterrors = np.sum(np.abs(txbits-ZFdatabinary))

###############################################################################
############################## GPU Processing #################################
###############################################################################

    # Set up platform
    devs, ctx, cq = init_ocl_runtime()

    ## Read pre-computed FFT twiddle factors
    fname = 'twiddle/w' + str(fftlen) + '.npy'
    tf = np.load(fname)

    ## Create device arrays. Use only one of the two equalizers since both have
    ## identical processing loads, project data can be obtained with either.
    d_eqfft = cl.array.to_device(cq, reorder(ZFfft))
    d_tf = cl.array.to_device(cq, tf)
    d_rxfrms = cl.array.to_device(cq, rxfrms)
    d_lfrm = cl.LocalMemory(fftlen*np.dtype('complex64').itemsize)

    # Create program object with GPU kernels
    with open("kernel.cl", "r") as fp:
        kernelsrc = fp.read()
        if PRINT_KERNELS == True:
            print kernelsrc

    # Configure system and build kernel
    #
    if fftlen not in kernelplan:
        print 'Error: invalid FFT size. Terminating'
        sys.exit(0)
    global_size = (kernelplan[fftlen][GLOBAL_DIM0], N)
    local_size = (kernelplan[fftlen][LOCAL_DIM0], 1)
    fftsizedef = kernelplan[fftlen][FFTSIZE_DEFINE]
    otherdefs = kernelplan[fftlen][OTHER_DEFINES]
    FFT_ONLY = True
    if FFT_ONLY == True:
        otherdefs += '\n#define FFT_ONLY 1'
    kernelsrc = fftsizedef + otherdefs + kernelsrc
    print 'gsize, lsize, fftsize, defs', global_size, local_size, fftsizedef,\
            otherdefs
    
    gpu = cl.Program(ctx, kernelsrc).build()

    # Write PTX code to file. Note: prg.binaries() returns a list object
    # containing the lines of the PTX souie
    ptx = gpu.binaries
    with open("gpu.ptx", "wb") as fp:
        fp.write(''.join(ptx))

    # Run GPU kernel. A single workgroup must handle one entire frame,
    # otherwise, intermediate results will need to be stored in global memory
    # to be shared by the workgroups. Parallelism -- distributing the
    # computations over as many compute units as possible -- is achieved by
    # processing together as many received frames as possible
    #
    gpuclocktm = time.clock()
    evt = gpu.r4ffteq(cq, global_size, local_size, d_rxfrms.data, d_tf.data,
            d_eqfft.data, d_lfrm)
    evt.wait()

    # Record wall and GPU times in msec
    gpuclocktm = 1e3*(time.clock()-gpuclocktm)
    gputm = 1e-6*(evt.profile.end-evt.profile.start)

    # Read Nth output frame from device to verify results.
    d_frmN = d_rxfrms[N-1].get()
    d_frmN_ordered = reorder(d_frmN)

    # For testing only: compute FFT using my implemenation, which matches
    # GPU's output order
    myDIFfft = r4DIFfft(rxfrms[N-1], tf)
    myDIFfft_ordered = reorder(myDIFfft)

    ## Print results
    np.set_printoptions(precision=3, suppress=True, linewidth=80,
            threshold=np.nan)
    print 'SciPy/Numpy Results:'
    print '  SNR: {}db'.format(snrdB)
    print '  Frame length: {} symbols'.format(txfrmlen)
    print '  ZF bit errors: {} out of {} ({:.2f}% BER)'.format(ZFbiterrors,
            txbits.size, (ZFbiterrors/np.float32(txbits.size))*100)
    print '  Avg FFT time {:.3f} ms'.format(npyavgffttm)
    print '  Avg iFFT time {:.3f} ms'.format(npyavgiffttm)
    print '  Avg Rx processing time per frame: {:.3f} ms'.format(npyavgfrmRxtm)
    print '  Total Rx processing time for {} frame(s) {:.3f} ms'.format(
            N, npyNfrmsRxtm)

    # For frame sizes > 1024, need to pass an absolute tolerance value for
    # allclose() that is smaller than the default 1e-08. Otherwise, double
    # precision needs to be used.
    print '\nGPU Processing Results:'
    print_device_info(devs)
    print 'Execution results:'
    print '  FFT size: {}'.format(fftlen)
    if FFT_ONLY == True:
        print '  FFT correct: ', np.allclose(myDIFfft, d_frmN, atol=1e-06)
        print '  FFT time for {} frame(s): {:.3f} ms'.format(N, gputm)
    else:
        print '  Zfoutfft correct: ', np.allclose(ZFoutfft, d_frmN_ordered,
            atol=1e-05)
        print '  EQ filtering time for {} frame(s): {:.3f} ms'.format(N, gputm)

    #temp = np.arange(1024, dtype=np.int32)
    #temp = reorder(temp)
    #fp = open('re1024.txt', 'w')
    #print >> fp, '\nReordered t[1024]:\n', temp
    #fp.close()

###############################################################################
############################ Done GPU Processing ##############################
###############################################################################

    # Plot various spectra for report
    plots_generated = False
    if PLOT_MAGNITUDE_SPECTRA == True:
        plot_magnitude_spectra(chfft, ZFfft, MMSEfft)
        plt.gcf()
        fname = 'mag' + str(snrdB) + 'dB.png'
        plt.savefig(fname)
        plots_generated = True

    if PLOT_POWER_SPECTRA == True:
        plot_power_spectra(txfrm, rxfrm, ZFout, MMSEout, fftlen)
        fname = 'psds' + str(snrdB) + 'dB.png'
        plt.savefig(fname)
        plots_generated = True

    # plt.show() call is blocking, so keep it as last statement in program 
    if plots_generated == True and SHOW_PLOTS == True:
        plt.show()
#------------------------------------------------------------------------------
# r4DIFfft(ifrm, tf)
#   This routine computes the radix-4 decimation-in-frequency (DIF) FFT of the
#   input data in ifrm[]. It uses the same algorithm implemented in the GPU and
#   is intended mainly for debugging and verification of GPU ops.
# Inputs:
#   ifrm: input frame array: ifrm[fftsize] (input array must be fftsize in len) 
#   tf: tf[N/4][4] twiddle factor array
# Returns:
#  fft[]: computed FFT
#------------------------------------------------------------------------------
def r4DIFfft(ifrm, tf, reorder_output=False):

    fft = ifrm.copy()

    # Compute FFT in n stages, where n is the power-of-four of the fftsize:
    # i.e., fft.size = N = 4^n.
    N2 = fft.size >> 2
    blks = 1
    blksize = fft.size

    # To avoid unnecessary multiplications with unity twiddle factors, leave
    # last stage to be done separately after this code block
    stages = (np.int32(np.log2(fft.size))>>1) - 1
    for st in xrange(stages):
        for blk in xrange(blks):
            # Get offset to current block
            i0 = blk*blksize
            # Compute N2 4-pt DFTs in this block
            for i in xrange(i0, i0+N2):
                r0 = fft[i] + fft[i+N2] + fft[i+2*N2] + fft[i+3*N2]
                r1 = fft[i] - fft[i+N2]*1j - fft[i+2*N2] + fft[i+3*N2]*1j
                r2 = fft[i] - fft[i+N2] + fft[i+2*N2] - fft[i+3*N2]
                r3 = fft[i] + fft[i+N2]*1j - fft[i+2*N2] - fft[i+3*N2]*1j
                # Multiply with twiddle factors and store in-place
                fft[i] = r0
                fft[i+N2] = r1 * tf[(i-i0)*blks][1]
                fft[i+2*N2] = r2 * tf[(i-i0)*blks][2]
                fft[i+3*N2] = r3 * tf[(i-i0)*blks][3]

        # Update parameters for next stage: blks*=4, blksize/=4, & N2/=4
        blks <<= 2
        blksize >>= 2
        N2 >>= 2

    # last stage: N/4 butterflies with no post-multiply by twiddle
    # factors
    for i in xrange(0, fft.size, 4):
        r0 = fft[i] + fft[i+1] + fft[i+2] + fft[i+3]
        r1 = fft[i] - fft[i+1]*1j - fft[i+2] + fft[i+3]*1j
        r2 = fft[i] - fft[i+1] + fft[i+2] - fft[i+3]
        r3 = fft[i] + fft[i+1]*1j - fft[i+2] - fft[i+3]*1j
        fft[i] = r0;
        fft[i+1] = r1;
        fft[i+2] = r2;
        fft[i+3] = r3;

    # Read in index map to permute order of FFT before display
    if reorder_output == True:
        fft = reorder(fft)

    return fft
#------------------------------------------------------------------------------
# reorder(frm)
#------------------------------------------------------------------------------
def reorder(frm):

    if frm.size not in [16, 64, 256, 1024, 4096]:
        print 'reorder(): error: frame length not supported. Exiting'
        sys.exit(0)

    f = 'idxmaps/n' + str(frm.size) + 'idxmap.npy'
    idxary = np.load(f)
    return frm[idxary]

#------------------------------------------------------------------------------
# QAMmodulate(txfrmlen, constell)
#   This function performs QAM modulation by generating random user data and
#   mapping to symbols in the given constellation.
# Inputs:
#   txfrmlen: Output frame length
#   constell: QAM constellation array
# Returns:
#   1. QAM-modulated data in a complex64 array 
#   2. User data in decimal and binary int8 arrays. (Binary bits must have an
#      unsigned integer type so that overflow is avoided in receiver when
#      computing bit errors. The operation abs(y-x) will overflow when
#      y = uint8(1) and x = uint8(0).
#------------------------------------------------------------------------------
def QAMmodulate(txfrmlen, constell):

    txdata = np.random.randint(constell.size, size=txfrmlen).astype(np.int8)
    txfrm = constell[txdata]

    #return (txdata, dec2bin(txdata, M=constell.size), txfrm)
    return txdata, txfrm

#------------------------------------------------------------------------------
# QAMdemodulate(rxfrm, constell)
#   This function performs QAM demodulation using the given constellation.
#
# Inputs:
#   rxfrm: Received frame array
#   constell: QAM constellation array
#
# Returns:
#   Demodulated bits in an int8 array. (Bits must have an unsigned integer type
#   so that overflow is avoided in receiver when computing bit errors. The
#   operation abs(y-x) will overflow when y = uint8(1) and x = uint8(0).
#------------------------------------------------------------------------------
def QAMdemodulate(rxfrm, constell):

    # A demodulated value is the index of the constellation symbol to which
    # it's closest
    demoddata = np.array(map(lambda i: np.argmin(np.abs(rxfrm[i]-constell)),
                    xrange(rxfrm.size))).astype(np.int8)

    return demoddata, dec2bin(demoddata, M=constell.size)

#------------------------------------------------------------------------------
# def dec2bin(ddata, M=16)
#   Converts an array of decimal symbol values into a binary stream
#
#   Inputs:
#       ddata: Integer array of decimal data
#       M: QAM constellation size. Either 16 (default) or 4
#
#   Returns:
#       Array of binary data with the same type as the input.
#------------------------------------------------------------------------------
def dec2bin(decdata, M):

    # Init # of symbols packed in a byte
    ns = 2
    if M == 4: ns = 4

    # Record input type
    dt = decdata.dtype
    decdata = decdata.astype(np.uint8)
    bindata = np.unpackbits(decdata).reshape(decdata.size, 8)
    arylist = np.hsplit(bindata, ns)

    # Return output in same type as input
    return arylist[ns-1].flatten().astype(dt)
#------------------------------------------------------------------------------
# plot_magnitude_spectra(chfft, ZFfft, MMSEfft)
#   This routine plots the magnitude spectra of the input frequency responses
#------------------------------------------------------------------------------
def plot_magnitude_spectra(chfft, ZFfft, MMSEfft):

    # Since channel is a real sequence, its magnitude response symetric, and we
    # need only plot half the data
    fftlen = chfft.size/2
    freqband = (np.arange(fftlen, dtype=np.float32)/fftlen)*1000

    # Create a new figure
    plt.figure()

    plt.title('Magnitude Responses of Channel and Equalizers') 
    #plt.title('Channel Magnitude Response') 
    plt.xlabel('Frequence (Hz)')
    plt.ylabel('Magnitude Response')
    plt.plot(freqband, np.abs(chfft[:fftlen]), 'b', label='Channel', lw=3)
    plt.plot(freqband, np.abs(ZFfft[:fftlen]), 'r', lw=2,
            label='ZF Equalizer')
    plt.plot(freqband, np.abs(MMSEfft[:fftlen]), 'g', lw=2, 
            label='MMSE Equalizer')
    plt.legend(loc=2, fontsize='medium')
    plt.grid(True)

    return

#------------------------------------------------------------------------------
# plot_power_spectra(qampsd, rxfrm, ZFout, MMSEout)
#   This routine computes and plots the power spectral densities for the
#   vectors passed in the input arguments
#------------------------------------------------------------------------------
def plot_power_spectra(txfrm, rxfrm, ZFout, MMSEout, fftlen):

    # Create new figure
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.05, top=0.95, hspace=0.40)

    # Since all input data are complex, the output PSDs will be two-sided, and
    # we need only examine one of them
    len = fftlen/2
    freqband = (np.arange(len, dtype=np.float32)/len)*1000

    f, qampsd = signal.welch(txfrm, nfft=fftlen)
    plt.subplot(411)
    plt.title('Spectrum of Transmitted QAM Symbols (Channel Input)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.plot(freqband, qampsd[:len])
    plt.grid(True)

    f, rxpsd = signal.welch(rxfrm, nfft=fftlen)
    plt.subplot(412)
    plt.title('Spectrum of Channel Output (Received Data)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.plot(freqband, rxpsd[:len])
    plt.grid(True)

    f, ZFoutpsd = signal.welch(ZFout, nfft=fftlen)
    plt.subplot(413)
    plt.title('Spectrum of ZF Equalizer Output')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.plot(freqband, ZFoutpsd[:len])
    plt.grid(True)

    f, MMSEoutpsd = signal.welch(MMSEout, nfft=fftlen)
    plt.subplot(414)
    plt.title('Spectrum of MMSE Equalizer Output')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.plot(freqband, MMSEoutpsd[:len])
    plt.grid(True)

    return
#------------------------------------------------------------------------------
# awgn(idata, snrdB, Es=None)
#   This routine computes Additive White Gaussian Noise (AWGN) based on the
#   desired output SNR and adds it to the input data.
#
# Input:
#   idata: Input frame data
#   snrdB: Desired output frame SNR in dB
#   Es: Average signal energy per symbol; (if None, value will be computed)
#
# Return:
#   odata = idata + noise 
#   N0: one-sided AWGN PSD (see notes below)
#
# Algorithm:
#   Relationship between Es/N0 and snrdB is: Es/N0 = 10log10(Tsym/Tsamp)+snrdB.
#   Since no oversampling is done (i.e., Tsym=Tsamp), Es/N0 = snrdB.
#
#       Calculations:
#           Es = sum(abs(idata)**2)/len(idata)
#           snrLinear = 10**(snrdB/10)
#           N0lp = Es/snrLinear
#           nreal = sqrt(N0lp/2)*randn(len(idata))
#           nimag = sqrt(N0lp/2)*randn(len(idata))*1.j
#           ndata = nreal + nimage
#           return idata+ndata, N0lp
#   Notes:
#       (1) For a WGN PSD of N0/2, the lowpass-equivalent PSD is 2*N0, i.e., 4x
#       the double-sided value (see Proakis 5th ed., p.81). To produce a
#       complex AWGN sequence, the real and imaginary parts are each generated
#       with variance=N0, since for i.i.d. complex samples xn = xn.r + xn.i*j,
#       var(xn) = var(xn.r) + var(xn.i). 
#
#       (2) EsN0 (dB) = EbN0 (dB) + 10log10(k), where k = # bits/symbol. k
#       is usually a combination of bits/modulation symbol + any redundancy
#       bits introduced by subsequent coding.
#------------------------------------------------------------------------------
def awgn(idata, snrdB, Es=None):

    # If not already done, calculate average energy
    if Es == None:
        Es = np.mean(np.abs(idata)**2)

    N0 = Es/10**(snrdB/10.0)/2.0
    ndata = np.sqrt(N0) * np.random.randn(idata.size).astype(idata.dtype) + \
            np.sqrt(N0) * np.random.randn(idata.size).astype(idata.dtype)*1.j

    # Get variance of generated samples. The value should be 2*N0 computed
    # above
    #print 'awgn(): LP N0 {:.2f}\n'.format(np.var(ndata))

    return (idata+ndata, N0)
#------------------------------------------------------------------------------
# dft16(x)
#   16-pt DFT by straight series computation. (For testing only)
#------------------------------------------------------------------------------
def dft16(x):

    a = np.arange(0.0, 16*np.pi/8, np.pi/8)
    y = np.zeros_like(x)
    for n in xrange(16):
        y += x[n]*(np.cos(n*a) - np.sin(n*a)*1j)

    return y
#------------------------------------------------------------------------------
# r4DIFdft(x, tf)
#   Testing 16-pt DIF
#------------------------------------------------------------------------------
def r4DIF16(x, tf):

    y = np.zeros_like(x)
    r = np.zeros(4, dtype=np.complex64)
    for i in xrange(4):
        k = i*4
        y[k]   = x[i] + x[i+4]    + x[i+8] + x[i+12]
        y[k+1] = x[i] - x[i+4]*1j - x[i+8] + x[i+12]*1j
        y[k+2] = x[i] - x[i+4]    + x[i+8] - x[i+12]
        y[k+3] = x[i] + x[i+4]*1j - x[i+8] - x[i+12]*1j
        y[k:k+4] *= tf[i]
    for k in xrange(4):
        r[0] = y[k] + y[k+4]    + y[k+8] + y[k+12]
        r[1] = y[k] - y[k+4]*1j - y[k+8] + y[k+12]*1j
        r[2] = y[k] - y[k+4]    + y[k+8] - y[k+12]
        r[3] = y[k] + y[k+4]*1j - y[k+8] - y[k+12]*1j
        y[k::4] = r

    return y

#------------------------------------------------------------------------------
# init_ocl_runtime(pltname='NVIDIA CUDA')
#   Sets up OpenCL runtime.
# Returns
#    Tuple: (context, queue)
#------------------------------------------------------------------------------
def init_ocl_runtime(pltname='NVIDIA CUDA'):

    platforms = cl.get_platforms()
    devs = None
    for platform in platforms:
        if platform.name == pltname:
            devs = platform.get_devices()

    # Set up command queue and enable GPU profiling
    context = cl.Context(devs)
    queue = cl.CommandQueue(context,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    #queue = cl.CommandQueue(context) 

    return (devs, context, queue)

#------------------------------------------------------------------------------
# print_device_info(devices)
#------------------------------------------------------------------------------
def print_device_info(devs):

    for device in devs:
        print 'Device Info:'
        print '  Name: {}'.format(device.name)
        print '  Max compute units: {}'.format(device.max_compute_units)
        print '  Max work group size: {}'.format(device.max_work_group_size)
        print '  Local mem size: {} bytes'.format(device.local_mem_size)

#------------------------------------------------------------------------------
# Define the entry point to the program
#------------------------------------------------------------------------------
if __name__ == '__main__':
    _main()
