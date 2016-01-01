/******************************************************************************
 * kernels.cl
 *  This file contains the GPU kernels used in the narrowband equalization
 *  project for ELEN E4750. 
 *
 * Author: Rashad Barghouti
 * UNI: rb3074
 * ELEN E4750, Fall 2015
 *****************************************************************************/

#include <pyopencl-complex.h>

#define FFTSIZE 1024U
#define FFT_ONLY 1

// Note: LOCAL_SIZE value below needs to agree with work-group size, which can
// be done at compile-time. Need to add code to do so later. 
//
#if FFTSIZE == 16U
    #define FFT_VALID 1U
    #define LOCAL_SIZE 1U
    #define N_STAGES 2U

#elif FFTSIZE == 64U
    #define FFT_VALID 1U
    #define LOCAL_SIZE 4U
    #define N_STAGES 3U

#elif FFTSIZE == 256U
    #define FFT_VALID 1U
    #define LOCAL_SIZE 16U
    #define N_STAGES 4U

#elif FFTSIZE == 1024U
    #define FFT_VALID 1
    #define LOCAL_SIZE 16U
    #define N_STAGES 5U

#elif FFTSIZE == 4096U
    #define FFT_VALID 1U
    #define LOCAL_SIZE 16U
    #define N_STAGES 6U
#else
    #define FFT_VALID 0U
    #define LOCAL_SIZE 1U
    #define N_STAGES 0U
#endif

#define TF_TBL_ROW_LEN 4U

// This kernel performs linear equalization filtering on input received frames.
// It takes as input an LxN matrix of L received frames, each in a
// (possibly zero-padded) array of length N. The discrete Fourier transform
// is computed for each frame (row) using a radix-4 decimation-in-frequency
// (DIF) fft algorithm.  The output is then multiplied by the equalizer filter
// coefficients, which are passed to the kernel by the host, followed by
// an inverse FFT step to obtain the demodulated frame (i.e., user data)
//
// The core operation performed in the DIF FFT is a 4-pt DFT, aka
// butterfly, shown here:
//
//      4-pt DFT (butterfly):
//      r0 = x[i] + x[i+istride]    + x[i+2*istride] + x[i+3*istride] 
//      r1 = x[i] - x[i+istride]*1j - x[i+2*istride] + x[i+3*istride]*1j
//      r2 = x[i] - x[i+istride]    + x[i+2*istride] - x[i+3*istride] 
//      r3 = x[i] + x[i+istride]*1j - x[i+2*istride] - x[i+3*istride]*1j
//
// The N-pt output is computed in M stages, where N = 4^M.  For all but the
// last stage, r1, r2, and r3 are multiplied by a twiddle factor of the form W
// = exp(-j*(2*pi/N)*k), where the value of k is derived from the specific
// input and output points involved in the calculation. 
//
// The FFT computations are done in-place, and the output is returned in the
// same input buffer. 
// 
__kernel void r4ffteq(global cfloat_t *rxfrms, global const cfloat_t *tf,
        global const cfloat_t *eqfft, local cfloat_t *lmemfrm) {

    global cfloat_t *frm = &rxfrms[get_global_id(1)*FFTSIZE];
    local cfloat_t *x = lmemfrm;

    uint tid,           // This thread's id
         istride,       // input access stride
         nblks,         // number of blocks in current stage
         blkid,         // id of current block of 4-pt DFTs
         blksize,       // number of input data points in current block
         dftid,         // id of current 4-pt DFT (butterfly) being computed
         ndfts,         // number of 4-pt DFTs to compute in current block
         i, n, k;
    cfloat_t r0, r1,
             r2, r3;    // registers used in butterfly computations
    cfloat_t fft[16];   // private mem buffer for computations in last stage

    // If FFTSIZE not in [16, 64, 256, 1024, 4096], exit
    if (!FFT_VALID)
        return;

    // Init vars
    tid = get_local_id(0);
    istride = FFTSIZE >> 2;
    nblks = 1U;
    blksize = FFTSIZE;
    ndfts = FFTSIZE >> 2;

    // Stage 1:
    // Process stage 1 independently, becasue its input is in global memory and
    // casting pointers between different memory spaces is not allowed in
    // OpenCL
#if FFTSIZE > 16U
    for (dftid = 0; dftid < ndfts; dftid += LOCAL_SIZE) {
        
        // Get linear offset to current 4-pt input vector 
        i = tid + dftid;

        // Compute butterfly 
        // Stride will be replaced by the compiler with FFTSIZE/4 constant
        r0 = frm[i] + frm[i+2*istride];
        r1 = frm[i] - frm[i+2*istride];
        r2 = frm[i+istride] + frm[i+3*istride];
        r3 = frm[i+istride] - frm[i+3*istride];
        
        // Multiply by twiddle factors and store output in local memory.
        //
        // n = twiddle factor table current row index
        //
        n = i*4U;
        x[i] = r0 + r2;
        r0 = r0 - r2;
        x[i+2*istride] = cfloat_mul(r0, tf[n+2]);
        r0.x = r1.x + r3.y;
        r0.y = r1.y - r3.x;
        r2.x = r1.x - r3.y;
        r2.y = r1.y + r3.x;
        x[i+istride] = cfloat_mul(r0, tf[n+1]);
        x[i+3*istride] = cfloat_mul(r2, tf[n+3]);
    }

    // Update execution parameters
    nblks <<= 2;
    blksize >>= 2;
    ndfts >>= 2;
    istride >>= 2;

    // Wait here for all threads to finish local memory writes
    barrier(CLK_LOCAL_MEM_FENCE);

#endif // Stage 1
    
    // Intermediate stages: done only for fft sizes > 64
    //
#if N_STAGES > 3U

    //__attribute__((opencl_unroll_hint));
    for (k = 0U; k < (N_STAGES-3U); k++) {
        for (blkid = 0; blkid < nblks; blkid++) {
            
            // Point to current block local memory
            x = lmemfrm+blkid*blksize;
            for (dftid = 0U; dftid < ndfts; dftid += LOCAL_SIZE) {

                i = tid + dftid;
                r0 = x[i] + x[i+2*istride];
                r1 = x[i] - x[i+2*istride];
                r2 = x[i+istride] + x[i+3*istride];
                r3 = x[i+istride] - x[i+3*istride];

                // Index tf table row multiply with twiddle factors
                n = i*nblks*4U;
                x[i] = r0 + r2;
                r0 = r0 - r2;
                x[i+2*istride] = cfloat_mul(r0, tf[n+2]);
                r0.x = r1.x + r3.y;
                r0.y = r1.y - r3.x;
                r2.x = r1.x - r3.y;
                r2.y = r1.y + r3.x;
                x[i+istride] = cfloat_mul(r0, tf[n+1]);
                x[i+3*istride] = cfloat_mul(r2, tf[n+3]);
            }
        }
        // Update parameters for next stage
        nblks <<= 2;
        blksize >>= 2;
        istride >>= 2;
        ndfts >>= 2;
    }
    // Wait here for all threads to finish local memory writes
    barrier(CLK_LOCAL_MEM_FENCE);

#endif // intermediate stages       

    // Do last two stages togther by computing a 16-pt DFT per thread instead
    // of a 4-pt one. There are three advantages to this approach: (1) Since in
    // the penultimate stage, the input data for one 16-pt DFT are
    // contiguous and aligned on a 128-byte boundary in memory, a thread can
    // access its entire input with a single memory transaction; (2) avoiding
    // unnecessary twiddle factor multiplications, which would happend if the
    // processing is done with the logic of the previous stages; and (3) no
    // intermediate store to memory is done.  The final output can be stored to
    // global memory straight from private registers.

    x = lmemfrm;
    for (blkid = 0U; blkid < nblks; blkid += LOCAL_SIZE) {
        
        i = (tid+blkid)*16U;

        // If FFTSIZE = 16, data has been processed in earlier stages and must
        // be read from global memory
#if FFTSIZE == 16U        
        r0 = frm[i] + frm[i+8];
        r1 = frm[i] - frm[i+8];
        r2 = frm[i+4] + frm[i+12];
        r3 = frm[i+4] - frm[i+12];
#else
        r0 = x[i] + x[i+8];
        r1 = x[i] - x[i+8];
        r2 = x[i+4] + x[i+12];
        r3 = x[i+4] - x[i+12];
#endif
        fft[0] = r0 + r2;
        fft[8] = r0 - r2;
        fft[4].x = r1.x + r3.y;
        fft[4].y = r1.y - r3.x;
        fft[12].x = r1.x - r3.y;
        fft[12].y = r1.y + r3.x;
        
        // Init index to twiddle factor table and finish remaining 3 rows
        //__attribute__((opencl_unroll_hint));
        for (k = 1; k < 4U; k++) {
#if FFTSIZE == 16
            r0 = frm[i+k] + frm[i+k+8];
            r1 = frm[i+k] - frm[i+k+8];
            r2 = frm[i+k+4] + frm[i+k+12];
            r3 = frm[i+k+4] - frm[i+k+12];
#else
            r0 = x[i+k] + x[i+k+8];
            r1 = x[i+k] - x[i+k+8];
            r2 = x[i+k+4] + x[i+k+12];
            r3 = x[i+k+4] - x[i+k+12];
#endif
            n = nblks*k*4U;
            fft[k] = r0 + r2;
            fft[k+8] = r0 - r2;
            fft[k+8] = cfloat_mul(fft[k+8], tf[n+2]);
            fft[k+4].x = r1.x + r3.y;
            fft[k+4].y = r1.y - r3.x;
            fft[k+4] = cfloat_mul(fft[k+4], tf[n+1]);
            fft[k+12].x = r1.x - r3.y;
            fft[k+12].y = r1.y + r3.x;
            fft[k+12] = cfloat_mul(fft[k+12], tf[n+3]);
        }
        // Do the column DFTs
        //__attribute__((opencl_unroll_hint));
        for (k = 0; k < 16U; k += 4U) {
            r0 = fft[k] + fft[k+2];
            r1 = fft[k] - fft[k+2];
            r2 = fft[k+1] + fft[k+3];
            r3 = fft[k+1] - fft[k+3];

#if FFT_ONLY == 1
            // Uncomment to do inverse fft.
            fft[k] = r0 + r2; 
            fft[k+1].x = r1.x + r3.y;
            fft[k+1].y = r1.y - r3.x;
            fft[k+2] = r0 - r2;
            fft[k+3].x = r1.x - r3.y;
            fft[k+3].y = r1.y + r3.x;
#else
            // Uncomment to do inverse fft.
            fft[k] = cfloat_mul((r0+r2), eqfft[i+k]); 
            fft[k+1].x = r1.x + r3.y;
            fft[k+1].y = r1.y - r3.x;
            fft[k+1] = cfloat_mul(fft[k+1], eqfft[i+k+1]);
            fft[k+2] = cfloat_mul((r0-r2), eqfft[i+k+2]);
            fft[k+3].x = r1.x - r3.y;
            fft[k+3].y = r1.y + r3.x;
            fft[k+3] = cfloat_mul(fft[k+3], eqfft[i+k+3]);
#endif
            
            // Delete this block to before finishing up inverse FFT
            //
            frm[i+k] = fft[k];
            frm[i+k+1] = fft[k+1];
            frm[i+k+2] = fft[k+2];
            frm[i+k+3] = fft[k+3];
            
        }
    }
    // ************************************************************************
    // ***************************** Inverse FFT ******************************
    // ************************************************************************
    
    //x = lmemfrm;
    //for (blkid = 0U; blkid < nblks; blkid++) {
    //    
    //    i = (tid+blkid)*16U;
    //    // First column in 16-pt DFT
    //    r0 = fft[0] + fft[2];
    //    r1 = fft[0] - fft[2];
    //    r2 = fft[1] + fft[3];
    //    r3 = fft[1] - fft[3];
    //    fft[0] = r0 + r2;
    //    fft[2] = r0 - r2;
    //    fft[1].x = r1.x + r3.y;
    //    fft[1].y = r1.y - r3.x;
    //    fft[3].x = r1.x - r3.y;
    //    fft[3].y = r1.y + r3.x;
    //    
    //    // Init twiddle factor table index and do remaining 3 columns
    //    n = 4U;
    //    //__attribute__((opencl_unroll_hint));
    //    for (k = 4; k < 16U; k+=4) {
    //        r0 = fft[k] + fft[k+2];
    //        r1 = fft[k] - fft[k+2];
    //        r2 = fft[k+1] + fft[k+3];
    //        r3 = fft[k+1] - fft[k+3];

    //        fft[k] = r0 + r2;
    //        
    //        // multiply with twiddle factor conjugate
    //        r0 = r0 - r2;
    //        fft[k+2].x = r0.x*tf[n+2].x + r.y*tf[n+2].y
    //        fft[k+2].y = r0.x*tf[n+2].y - r.x*tf[n+2].y
    //        r0.x = r1.x + r3.y;
    //        r0.y = r1.y - r3.x;
    //        fft[k+1].x = r0.x*tf[n+1].x + r0.y*tf[n+1].y
    //        fft[k+1].y = r0.x*tf[n+1].y - r0.x*tf[n+1].y
    //        r2.x = r1.x - r3.y;
    //        r2.y = r1.y + r3.x;
    //        fft[k+3].x = r2.x*tf[n+3].x + r2.y*tf[n+3].y
    //        fft[k+3].y = r2.x*tf[n+3].y - r2.x*tf[n+3].y

    //        n *= 2;
    //    }

    //    // Do the row DFTs
    //    //__attribute__((opencl_unroll_hint));
    //    for (k = 0; k < 4U; k++) {
    //        r0 = fft[k] + fft[k+8];
    //        r1 = fft[k] - fft[k+8];
    //        r2 = fft[k+4] + fft[k+12];
    //        r3 = fft[k+4] - fft[k+12];
    //        x[i+k] = r0 + r2; 
    //        x[i+k+4].x = r1.x + r3.y;
    //        x[i+k+4].y = r1.y - r3.x;
    //        x[i+k+8] = r0 - r2;
    //        x[i+k+12].x = r1.x - r3.y;
    //        x[i+k+12].y = r1.y + r3.x;
    //    }
    //}
    //barrier(CLK_LOCAL_MEM_FENCE);

    ////__attribute__((opencl_unroll_hint));
    //for (k = 0U; k < (N_STAGES-2U); k++) {
    //    nblks >>= 2;
    //    blksize <<= 2;
    //    istride <<= 2;
    //    ndfts <<= 2;
    //    for (blkid = 0; blkid < nblks; blkid++) {
    //        
    //        // Point to current block local memory
    //        x = lmemfrm+blkid*blksize;
    //        for (dftid = 0U; dftid < ndfts; dftid += LOCAL_SIZE) {

    //            i = tid + dftid;
    //            r0 = x[i] + x[i+2*istride];
    //            r1 = x[i] - x[i+2*istride];
    //            r2 = x[i+istride] + x[i+3*istride];
    //            r3 = x[i+istride] - x[i+3*istride];

    //            // Index tf table row multiply with twiddle factors
    //            n = i*nblks*4U;
    //            x[i] = r0 + r2;
    //            r0 = r0 - r2;
    //            x[i+2*istride].x = r0.x*tf[n+2].x + r0.y*tf[n+2].y
    //            x[i+2*istride].y = r0.x*tf[n+2].y - r0.x*tf[n+2].y
    //            r0.x = r1.x + r3.y;
    //            r0.y = r1.y - r3.x;
    //            x[i+istride].x = r0.x*tf[n+1].x + r0.y*tf[n+1].y
    //            x[i+istride].y = r0.x*tf[n+1].y - r0.x*tf[n+1].y
    //            r2.x = r1.x - r3.y;
    //            r2.y = r1.y + r3.x;
    //            x[i+3*istride].x = r2.x*tf[n+3].x + r2.y*tf[n+3].y
    //            x[i+3*istride].y = r2.x*tf[n+3].y - r2.x*tf[n+3].y
    //        }
    //    }
    //barrier(CLK_LOCAL_MEM_FENCE);
    //}
    // Wait here for all threads to finish local memory writes
}
