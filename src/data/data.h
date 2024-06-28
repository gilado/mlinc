/* Copyright (c) 2023-2024 Gilad Odinak */
/* Data processing functions              */
#ifndef _DATA_H
#define _DATA_H
#include "array.h"

/* The variables D, S, K, T are used throughout:
 * D: number of dimensions of input vectors *including bias*
 * S: size of hidden layer
 * K: number of output classes
 * T: number of time steps
 */

#define MAX_FILES     10000
#define MAX_SAMPLES    1000 // Per file

typedef struct sequence_s {
    int num_samples;
    SAMPLE *samples;
} SEQUENCE;

#define MAX_VECTORS 2000000 // Total training set

/* Parses textual sample infomation in line and populates sample.
 * Return 0 if line parsed successfully; otherwise returns -1.
 */
int parseline(char *line, int lcnt, SAMPLE* restrict sample);

/* Loads samples from feature files. Each files contains a sequences of
 * phonemes. Each phoneme consist of multiple frames of features. Each
 * frame spans a fixed time period (e.g. 10 milliseconds).
 * The samples are stored in sequences array (whose sie is max_sequences)
 * Returns the actual number of sequences stored in the array.
 * Returns -1 on error.
 */
int load_data(const char* listfile, SEQUENCE* sequences, int max_sequences);

/* Creates feature vectors and corresponding label vectors from sequence data.
 * Returns the actual number of vectors stored in the arrays.
 * Returns -1 on error.
 * Note that this function shuffles the entries in sequences array in place.
 * x: matrix with dimensions max_vectors x D (D includes bias)
 * y: matrix with dimensions max_vectors x K
 * sequences: a list with length of num_sequencs
 * seq_len: the number of vectors in that sequence
 * Returns the actual number of vectors stored in x and y.
 */
int prepare_data(fArr2D x/*[max_vectors][D]*/,
                 int D,
                 fArr2D y/*[max_vectors][K]*/,
                 int K, 
                 int max_vectors, 
                 SEQUENCE* sequences, 
                 int num_sequences,
                 int* len_seq/*[num_sequences]*/
);

/* Calculates element wise mean and standard deviation for all feature vectors
 * in the data set passed in x[][], and returns that mean and stddev arrays.
 * Excludes the last column in x[][] (the bias)
 * num_vectors: number of vectors in x.
 * D: number of dimensions of input vectors in x (including bias)
 */
void summary_stats(const fArr2D x /*[num_vectors][D]*/, int num_vectors,
                           fVec mean/*[D-1]*/, fVec stddev/*[D-1]*/, int D);

/* Normalizes all feature frames in the sequences data set in place.
 * Excludes the last column in x[][] (the bias)
 * num_vectors: number of vectors in x.
 * D: number of dimensions of input vectors in x (including bias)
 */
void normalize_data(fArr2D x/*[num_vectors][D]*/, int num_vectors,
               const fVec mean/*[D-1]*/, const fVec stddev/*[D-1]*/, int D);

/* Calculates and returns class weights based on class frequencies 
 * in the passed in label data.
 * num_vectors: number of vectors in y.
 * K: number of output classes
 */
void class_stats(fArr2D y/*[num_vectors][K]*/, int num_vectors,
                                           fVec class_weights/*[K]*/, int K);

#endif
