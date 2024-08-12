/* Copyright (c) 2023-2024 Gilad Odinak */
/* Embedding neural network layer data structures and functions */
#ifndef EMBEDDING_H
#define EMBEDDING_H
#include "array.h"

typedef struct embedding_s {
  int D;       /* Vocabulary size            */
  int S;       /* Output size (== E)         */
  int B;       /* Batch size                 */
  int M;       /* Context length             */
  int E;       /* Embedding dimension        */
  int padinx;  /* Pad index, -1 if not used  */
  fArr2D h;    /* Hidden State matrix [B][S] */
  fArr2D Wx;   /* Weights matrix [D][E]      */
} EMBEDDING;

/* Creates an embedding layer.
 *
 * Parameters:
 *   embedding_dim - Dimension of token embedding vectors
 *   context_len   - Number of token indices in a context
 *   padinx        - Index value of pad token, -1 if not used
 *
 *
 * Returns:
 *   Pointer to an EMBEDDING layer.
 *
 * Notes:
 *   - Contexts shorter than context_length are padded with blank (pad index)
 *   - The embedding needs to be further intialized using embedding_init()
 *     before it can be used.
 */
EMBEDDING* embedding_create(int embedding_dim, int context_len, int padinx);

/* Initializes an embedding layer created by embedding_create().
 *
 * Parameters:
 *   vocab_size - Number of vocabulary tokens (including blank, if any)
 *   batch_size - Number of input contexts processed simultaneously
 *
 * Notes:
 *   The network's weights are initialized using uniform distribution 
 */
void embedding_init(EMBEDDING* l, int vocab_size, int batch_size);

/* Frees the memory allocated by embedding_create() / embedding_init()
 * 
 * Parameters:
 *   l - Pointer to the neural network to be freed
 */
void embedding_free(EMBEDDING* l);

/* Resets the network hidden state.
 * 
 * Parameters:
 *   l - Pointer to the EMBEDDING neural network layer to be reset
 */
void embedding_reset(EMBEDDING* l);

/* Performs embedding layer training/prediction's forward pass.
 *
 * Parameters:
 *   l   - Pointer to the embedding layer's data
 *   X   - An array of input indices representing one hot encoded vectors
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Returns:
 *   Pointer to the predicted values, a 3D array h[B][M][E], which can be
 *   viewed as a flattened 2D array h[B][S], where S = M * E
 *
 * Note that in a multi-layered neural network, after the first layer
 * X is the output of a previous layer.
 */
static inline fArr2D embedding_forward(EMBEDDING* restrict l, 
                                       fArr2D restrict X_/*[B][M]*/, int lyr)
{
    (void) lyr;
    typedef float (*ArrBM)[l->M];
    typedef float (*ArrBS)[l->S];
    typedef float (*ArrDE)[l->E];
    ArrBM X = (ArrBM) X_;
    ArrBS h = (ArrBS) l->h;
    ArrDE Wx = (ArrDE) l->Wx;
    /* X[B][M] => x[B][M][D] where x[B][M] is one-hot encoded */
    /* h = x @ Wx  => sum context vectors */
    fltclr(h,l->B * l->S);
    for (int i = 0; i < l->B; i++)
        for (int j = 0; j < l->M; j++)
            for (int k = 0; k <l->E; k++)
                h[i][j] += Wx[(int)X[i][j]][k];
    return l->h;
}

/* Performs embedding layer training's backward pass.
 *
 * Parameters:
 *   l   - Pointer to the embedding layer's data
 *   dy  - The output vector gradient of embedding_create's units dimension
 *   X   - An array of input indices representing one hot encoded vectors
 *   lyr - The ordinal number of this layer in a model (not used)
 *
 * Calculates the weight matrix gradients with respect to the weights 
 * and adds them to the matrix gWx
 *
 * Calculates the input vector gradient and returns it in dx, if dx is not NULL
 *
 * Note that in a multi-layered neural network, except the last layer,
 * dy is the gradient of the previous layer's input (dx), thus, the dimension
 * of dx (this layer's D) must equal the dimension of the previous layer's
 * dy (previous layer's S).
 */
static inline void embedding_backward(EMBEDDING* restrict l, 
                                  const fArr2D restrict dy_/*[B][S]*/, 
                                  const fArr2D restrict X_/*[B][M]*/,
                                  fArr2D restrict gWx_/*[D][E]*/,
                                  fArr2D restrict dx_/*[B][M]*/, int lyr)
{
    (void) lyr;
    typedef float (*ArrBM)[l->M];
    typedef float (*ArrBS)[l->S];
    typedef float (*ArrDE)[l->E];
    ArrBS dy = (ArrBS) dy_;
    ArrBM X = (ArrBM) X_;
    ArrDE gWx = (ArrDE) gWx_;
    ArrBM dx = (ArrBM) dx_;

    /* X[B][M] => x[B][M][D] where x[B][M] is one-hot encoded */
    /* Gradient with respect to weights: gWx = x.T @ dy */
    fltclr(gWx,l->D * l->E);    
    for (int i = 0; i < l->B; i++) {
        for (int j = 0; j < l->M; j++) {
            int xij = (int) X[i][j];
            if (xij != l->padinx) {
                for (int k = 0; k < l->E; k++)
                    gWx[xij][k] += dy[i][k] / l->M;
            }
        }
    }
    if (dx != NULL) { /* dx = (dy @ Wx.T) */
        fltclr(dx,l->B * l->M);
        for (int i = 0; i < l->B; i++) {
            for (int j = 0; j < l->M; j++) {
                for (int k = 0; k < l->E; k++)
                    dx[i][j] += dy[i][k] / l->M;
            }
        }
    }
}
#endif
