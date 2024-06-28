/* Copyright (c) 2023-2024 Gilad Odinak */
/* Function to creates new delta features from exsiting features */
#include "delta.h"

/* Calculates feature deltas.
 *
 * Feature vectors are stored in the passed in array x. Each feature vector 
 * is stored in a separate row, starting at offset soff, and occupying fcnt
 * elements. This function calculates deltas for the features, and for each
 * feature vector stores its deltas in the same row starting at column doff.
 *
 * Parameters:
 *   x     - Array of vectors (input/output)
 *   M     - Number of rows in x
 *   N     - Number of columns in x
 *   soff  - column offset in x where features are stored (input)
 *   doff  - column offset in x where the deltas whould be stored (output)
 *   fcnt  - Number of features (applies to features and deltas)
 *   wsize - Delta window size
 *
 * Returns:
 *   Calculated deltas, stored back into x.
 *
 * Note:
 *   This function can be used to calculate delta-deltas by setting soff
 *   to the column in x where previously calculated deltas have been stored
 *   and setting doff to the clumn where the delta-deltas will be stored.
 *
 * Reference:
 *   http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#deltas-and-delta-deltas
 * 
 */
void calculate_deltas(float* x_, int M, int N, 
                      int soff, int doff, int fcnt, int wsize) 
{
    typedef float (*ArrMN)[N];
    ArrMN x = (ArrMN) x_;

    int denominator = 0;
    for (int n = 1; n <= wsize; n++)
        denominator += n * n;    
    denominator *= 2;

    for (int t = 0; t < M; t++) {
        for (int f = 0; f < fcnt; f++) {
            float numerator = 0.0;
            for (int n = 1; n <= wsize; n++) {
                if (t + n < M)
                    numerator += n * x[t + n][soff + f];
                if (t - n >= 0)
                    numerator -= n * x[t - n][soff + f];
            }
            x[t][doff + f] = numerator / denominator;
        }
    }
}

