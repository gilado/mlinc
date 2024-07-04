/* Copyright (c) 2023-2024 Gilad Odinak */
#include <math.h>

#include "mem.h"
#include "float.h"
#include "array.h"

/* Calculates element wise mean and standard deviation for all feature vectors
 * in the data set passed in x[][], and returns that mean and stddev arrays.
 * The calculation excludes the last column in x[][] (the bias)
 * num_vectors: number of vectors in x.
 * D: number of dimensions of input vectors in x (including bias)
 */
void summary_stats(const fArr2D x_ /*[num_vectors][D]*/, int num_vectors,
                         fVec mean_/*[D-1]*/, fVec stddev_/*[D-1]*/, int D)
{
    typedef float (*ArrND)[D];
    ArrND x = (ArrND) x_;

    if (num_vectors <= 0)
        return;
    
    typedef float (*VecDx);
    VecDx mean = (VecDx) mean_;
    VecDx stddev = (VecDx) stddev_;
    // Notice Dx denotes last column (the bias) is excluded
    float sum[D - 1]; fltclr(sum,D - 1);
    float var[D - 1]; fltclr(var,D - 1);
    for (int i = 0; i < num_vectors; i++) {
        for (int j = 0; j < D - 1; j++)
            sum[j] += x[i][j];
    }
    for (int j = 0; j < D - 1; j++)
        mean[j] = sum[j] / num_vectors;

    for (int i = 0; i < num_vectors; i++) {
        for (int j = 0; j < D - 1; j++) {
            float val = x[i][j] - mean[j];
            var[j] += val * val;
        }
    }
    for (int j = 0; j < D - 1; j++)
        stddev[j] = sqrt(var[j] / num_vectors);
}

/* Normalizes all feature frames in the sequences data set in place.
 * The normalization Excludes the last column in x[][] (the bias)
 * num_vectors: number of vectors in x.
 * D: number of dimensions of input vectors in x (including bias)
 */
void normalize_data(fArr2D x_/*[num_vectors][D]*/, int num_vectors,
         const fVec mean_/*[D - 1]*/, const fVec stddev_/*[D - 1]*/, int D)
{
    typedef float (*ArrND)[D];
    ArrND x = (ArrND) x_;
    // Notice Dx denotes last column (the bias) is excluded   
    typedef float (*VecDx);
    VecDx mean = (VecDx) mean_;
    VecDx stddev = (VecDx) stddev_;

    for (int i = 0; i < num_vectors; i++) {
        for (int j = 0; j < D - 1; j++)
            if (stddev[j] > 0.0)
                x[i][j] = (x[i][j] - mean[j]) / (stddev[j]);
            else
                x[i][j] = 0.0;
    }
}

/* Calculates and returns class weights based on class frequencies 
 * in the passed in label data.
 * num_vectors: number of vectors in y.
 * K: number of output classes
 */
void class_stats(fArr2D y_/*[num_vectors][K]*/, int num_vectors,
                                          fVec class_weights_/*[K]*/, int K)
{
    typedef float (*ArrNK)[K];
    ArrNK y = (ArrNK) y_;
    typedef float (*VecK);
    VecK class_weights = (VecK) class_weights_;

    fltclr(class_weights,K);
    // Obtain class frequencies (how many samples per class)
    // Since y[] is one-hot encoded, just sum all
    for (int i = 0; i < num_vectors; i++) {
        for (int j = 0; j < K; j++)
            class_weights[j] += y[i][j];
    }
    // Calculate class weights from frequencies
    for (int i = 0; i < K; i++) {
        if (class_weights[i] > 0)
            class_weights[i] = num_vectors / class_weights[i];
    }
    // Normalize
    float sum = 0.0;
    for (int i = 0; i < K; i++)
        sum += class_weights[i];
    for (int i = 0; i < K; i++)
        class_weights[i] = K * class_weights[i] / sum;

    // Override
    //for (int i = 0; i < K; i++)
    //    class_weights[i] = 1.0;
}
