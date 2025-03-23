/* Copyright (c) 2023-2024 Gilad Odinak                   */
/* Approximate Nearest Neighbors structures and functions */

#ifndef ANNOY_H
#define ANNOY_H
#include "float.h"
#include "array.h"

/* Annoy search tree node is referenced by an index
 * of the node in an array of pre-allocated tree nodes.
 */
typedef struct annoy_node_s ANNOY_NODE;
typedef int ANNOY_NODE_IX;

typedef struct annoy_s {
    fArr2D data;          /* [N][D], Array of N vectors of D dimensions      */
    int num_vec;          /* Number of vectors in data [N]                   */
    int vec_dim;          /* Number of elements in each vector [D}           */
    int num_trees;        /* Number of trees                                 */
    int leaf_size;        /* Maximum number of data indices in a leaf        */
    int num_nodes;        /* Allocated number of tree nodes (for all trees)  */
    int num_used;         /* Number of tree nodes used                       */
    int hpv_cnt;          /* Number of hyperplanes == non-leaf nodes         */
    ANNOY_NODE* nodes;    /* Array of tree nodes [num_nodes]                 */
    ANNOY_NODE_IX* root;  /* Array of trees [num_trees]                      */
    int cos_sim_cnt;      /* Counts cosine similarity calls in one search    */
} ANNOY;

/* Leaf nodes directly store indices of vectors that belong in the leaf.
 * ANNOY_LDS is the number of indices that fits in a node
 */
#define ANNOY_LDS ((int)((2*sizeof(int)+2*sizeof(ANNOY_NODE_IX))/sizeof(int)))

struct annoy_node_s {
    int vcnt;                   /* Number of vectors in this leaf or tree    */
    union {                     /* This node's data                          */
        int data[ANNOY_LDS];    /* Leaf's vector indices (vcnt <= ANNOY_LDS) */
        struct {                /* This tree node (vnct > ANNOY_LDS)         */
            int split[2];       /* Indices of vectors defining the hyperplane*/
            ANNOY_NODE_IX left; /* Sub tree of vectors 'left' of hyperplane  */
            ANNOY_NODE_IX right;/* Sub tree of vectors 'right' of hyperplane */
        };
    };
};

/* Creates an Approximate Nearest Neighbors search tree.
 *
 * Parameters:
 *   data      - An array of vectors to be searched.
 *   num_vec   - Number of data vectors.
 *   vec_dim   - Vectors' dimension, number of elements in each vector.
 *   num_trees - Number of search trees.
 *
 * Returns:
 *   A pointer to Annoy search tree.
 *
 * Notes:
 *   Annoy only keeps a reference to the data, it does not make a copy of it.
 *   The data used by Annoy must not change until after calling annoy_free().
 *   Internally Annoy uses the row indices of data to retrieve vectos.
 *   Annoy uses vectors cosine similarity to compare vectors.
 */
ANNOY* annoy_create(const fArr2D data, int num_vec, int vec_dim, int num_trees);

/* Finds vectors most similar to a given vector.
 *
 * Parameters: 
 *   annoy      - Annoy search tree.
 *   query      - Vector to search for; its dimension must be the same 
 *                as the dimension of the data vectors used to create
 *                the Annoy search tree.
 *   search_q   - quality vs time; 0: best quality 1: fastest
 *   similar    - Pointer to integer array that receives indices of vectors
 *                in data array passed to annoy_create() that are most similar
 *                to the query vector, in descending similarity order.
 *   similarity - Pointer to optional float array that receives the cosine
 *                similarity values of the vectors, whose corresponding indices
 *                in data are returned in similar, and the query vector.
 *   topn       - The number of entries in similar and similarity arrays.
 *
 * Returns:
 *   The actual number of entries retruend similar, similarity arrays, 
 *   which may be less than topn.
 */
int annoy_most_similar(ANNOY* annoy, const fVec query, float search_q,
                                     int* similar, float* similarity, int topn);

/* Frees the memory allocated by annoy_create()
 */
void annoy_free(ANNOY* annoy);                                

#endif
