/* Copyright (c) 2023-2024 Gilad Odinak  */
/* Function to find vectors in a dataset */

#ifndef FINDSIM_H
#define FINDSIM_H
#include "array.h"

/* Finds the vectors in a dataset that are most similar to a given vector.
 *
 * This function computes the cosine similarity between a query vector and 
 * each vector in the dataset. It identifies the top topn most similar vectors
 * and stores their indices and similarity scoresi in the passed in arrays.
 *
 * Arguments:
 *  data       - 2D array of shape [num_vec][vec_dim] representing the dataset.
 *  num_vec    - Number of vectors in the dataset.
 *  vec_dim    - Dimensionality of each vector.
 *  query      - 1D array representing the query vector.
 *  similar    - Output array where indices of the top "topn" similar vectors
 *               in the dataset will be stored.
 *  similarity - Optional output array where similarity scores of the top
 *               "topn" similar vectors will be stored.
 *  topn       - Number of top similar vectors to find.
 *
 * Returns:
 *  The number of vectors found, which will be "topn" unless the dataset 
 *  contains fewer than "topn" vectors.
 */
int find_most_similar(const fArr2D data_, int num_vec, int vec_dim,
                 const fVec query_, int* similar, float* similarity, int topn);

#endif
