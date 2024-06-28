/* Copyright (c) 2023-2024 Gilad Odinak */
/* Data processing functions              */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <assert.h>
#include <math.h>

#include "random.h"
#include "array.h"
#include "sample.h"
#include "data.h"

/* The variables D, S, K, T are used throughout:
 * D: number of dimensions of input vectors *including bias*
 * S: size of hidden layer
 * K: number of output classes
 * T: number of time steps
 */

/* Creates feature vectors and corresponding label vectors from sequence data.
 * Returns the actual number of vectors stored in the arrays.
 * Returns -1 on error.
 * Note that this function shuffles the entries in sequences array in place.
 * x: matrix with dimensions max_vectors x D  (D includes bias)
 * y: matrix with dimensions max_vectors x K
 * sequences: a list with length of num_sequencs
 * seq_len: the number of vectors in that sequence
 * Returns the actual number of vectors stored in x and y.
 */
int prepare_data(fArr2D x_/*[max_vectors][D]*/,
                 int D,
                 fArr2D y_/*[max_vectors][K]*/,
                 int K, 
                 int max_vectors, 
                 SEQUENCE* sequences, 
                 int num_sequences,
                 int* seq_len/*[num_sequences]*/)
{
    typedef float (*ArrND)[D];
    ArrND x = (ArrND) x_;
    typedef float (*ArrNK)[K];
    ArrNK  y = (ArrNK)  y_;
    /* Shuffle sequences */
    for (int k = 0; k < 3; k++) {
        for (int i = num_sequences - 1; i > 0; i--) {
            int j = (int) urand(0.0,1.0 + i);
            SEQUENCE tmp = sequences[i];
            sequences[i] = sequences[j];
            sequences[j] = tmp;
        }
    }
    // Process all sequences
    int nvec = 0; // total number of vectors
    for (int seqinx = 0; seqinx < num_sequences; seqinx++) {
        seq_len[seqinx] = 0;
        SAMPLE *samples = sequences[seqinx].samples;
        int nsamples = sequences[seqinx].num_samples;
        int vinx = 0; // index of parameter in vector
        int sinx = 0; // sample index
        int finx = 0; // index of frame in sample
        int pinx = 0; // index of parameter in frame[finx]
        // Process all samples in a sequence
        while (sinx < nsamples) {
            int nfrm = samples[sinx].num_frames;
            if (nfrm == 0) { // sample has no data
                fprintf(stderr,"Sample has no frames, and cannot be adjusted\n");
                sinx++;
                continue;
            }
            // Pack a sample frames into one or more vectors                
            x[nvec][vinx++] = samples[sinx].features[finx][pinx++]; 
            if (pinx >= FRAME_SIZE) { // move to next frame
                pinx = 0;
                finx++; // next frame
            }
            if (pinx == 0) { // complete vector
                assert(vinx == D - 1);
                x[nvec][vinx] = 1.0;   // add bias
                for (int i = 0; i < NUM_CLASSES; i++)
                    y[nvec][i] = samples[sinx].expected_output[i];
                vinx = 0;
                nvec++; // next vector
                seq_len[seqinx]++;
                if (nvec >= max_vectors)
                    break;
            }
            if (finx >= nfrm) { // move to next sample   
                finx = 0;
                sinx++;
            }
        }
        if (nvec >= max_vectors) {
            fprintf(stderr,"reached %d vectors - rest ignored\n",nvec);
            break;
        }
    }
    return nvec;
}


