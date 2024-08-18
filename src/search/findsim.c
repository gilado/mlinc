/* Copyright (c) 2023-2024 Gilad Odinak  */
/* Function to find vectors in a dataset */
#include <stdlib.h>
#include "cossim.h"
#include "findsim.h"

typedef struct {
    int data_ix;  /* Index of a data vector */
    float cossim; /* Cosine similarity of this vector to the query vector */
} COSSIM;

/* Sorts by similarity, then by vector index in ascending order */
static int qsort_compare_asc(const void *a_, const void *b_)
{
    COSSIM* a = (COSSIM*)a_;
    COSSIM* b = (COSSIM*)b_;
    if (a->cossim < b->cossim) return -1;
    if (a->cossim > b->cossim) return 1;
    if (a->data_ix < b->data_ix) return -1;
    if (a->data_ix > b->data_ix) return 1;
    return 0;
}

/* Sorts entries by similarity order, then by vector index order.
 * In the sorted array duplicate entries will be adjecant, so they
 * can be easily removed.
 */
static inline void sort_asc(COSSIM* sim, int cnt)
{
    qsort(sim,cnt,sizeof(COSSIM),qsort_compare_asc);
}

/* Sorts by similarity, then by vector index in descending order */
static int qsort_compare_desc(const void *a_, const void *b_)
{
    COSSIM* a = (COSSIM*)a_;
    COSSIM* b = (COSSIM*)b_;
    if (a->cossim < b->cossim) return 1;
    if (a->cossim > b->cossim) return -1;
    if (a->data_ix < b->data_ix) return 1;
    if (a->data_ix > b->data_ix) return -1;
    return 0;
}

/* Sorts entries by similarity order, then by vector index order.
 * In the sorted array duplicate entries will be adjecant, so they
 * can be easily removed.
 */
static inline void sort_desc(COSSIM* sim, int cnt)
{
    qsort(sim,cnt,sizeof(COSSIM),qsort_compare_desc);
}

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
 *
 * The function uses a COSSIM array to track the most similar vectors.
 * It sorts this array first in ascending order to maintain the smallest 
 * similarity at index 0. If a vector has a higher similarity than the 
 * smallest one in the array, it replaces the smallest and the array is 
 * re-sorted. After processing all vectors, the COSSIM array is sorted in 
 * descending order, and the indices and similarity scores of the top "topn"
 * vectors are copied to the "similar" and "similarity" arrays.
 */
int find_most_similar(const fArr2D data_, int num_vec, int vec_dim,
                  const fVec query_, int* similar, float* similarity, int topn)
{
    typedef float (*ArrND)[vec_dim];
    ArrND data = (ArrND) data_;
    typedef float (*VecD);
    VecD query = (VecD) query_;
    COSSIM sim[topn];

    int cnt = 0;
    for (int i = 0; i < num_vec; i++) {
        float cossim = cosine_similarity(query,data[i],vec_dim);
        if (cnt < topn) {
            sim[cnt].data_ix = i;
            sim[cnt].cossim = cossim;
            cnt++;
            sort_asc(sim,cnt);
        }
        else
        if (cossim > sim[0].cossim) {
            sim[0].data_ix = i;
            sim[0].cossim = cossim;
            sort_asc(sim,cnt);
        }
    }
    sort_desc(sim,cnt);
    for (int i = 0; i < cnt && i < topn; i++)
        similar[i] = sim[i].data_ix;
    for (int i = 0; i < cnt && i < topn; i++)
        similarity[i] = sim[i].cossim;
    return cnt;
}
