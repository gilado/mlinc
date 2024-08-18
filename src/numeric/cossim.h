/* Copyright (c) 2023-2024 Gilad Odinak */
/* Vector cosine similarity function    */
#ifndef COSSIM_H
#define COSSIM_H
#include "array.h"
#include "norm.h"

/* cosine_similarity - Compute the cosine similarity between two vectors
 *
 * This function calculates the cosine similarity between two vectors 
 * v1 and v2 of dimension dim. Cosine similarity is a measure of the cosine
 * of the angle  between two non-zero vectors in a multi-dimensional space.
 *
 * Parameters:
 *   v1  - Pointer to the first vector
 *   v2  - Pointer to the second vector
 *   dim - The dimensionality of the vectors (number of elements)
 *
 * Returns:
 *   The cosine similarity as a float between -1 and 1. 
 *
 * Notes:
 *   - If v1 or v2 is NULL, the function returns 0.
 *   - If either vector has a norm of zero, indicating a zero vector,
 *     the function returns 0.
 *   - The function assumes that the vectors are of the same dimension.
 */
static inline float cosine_similarity(
    const fVec restrict v1, 
    const fVec restrict v2,
    int dim)
{
    if (v1 == NULL || v2 == NULL) return 0;
    float norm_v1 = vecnorm(v1,dim);
    if (norm_v1 == 0) return 0;
    float norm_v2 = vecnorm(v2,dim);
    if (norm_v2 == 0) return 0;
    float dot_product = 0.0;
    for (int i = 0; i < dim; i++)
        dot_product += v1[i] * v2[i];
    return dot_product / (norm_v1 * norm_v2);
}

#endif
