/* Copyright (c) 2023-2024 Gilad Odinak */
/* Embedding neural network layer functions */
#include "mem.h"
#include "array.h"
#include "random.h"
#include "embedding.h"

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
EMBEDDING* embedding_create(int embedding_dim, int context_len, int padinx)
{
    EMBEDDING* l = allocmem(1,1,EMBEDDING);
    l->E = embedding_dim;
    l->M = context_len;
    l->S = embedding_dim;
    l->padinx = padinx;
    return l;
}

/* Initializes an embedding layer created by embedding_create().
 *
 * Parameters:
 *   vocab_size - Number vocabulary tokens (including blank, if any)
 *   batch_size - Number of input contexts processed simultaneously
 *
 * Notes:
 *   The network's weights are initialized using uniform distribution 
 */
void embedding_init(EMBEDDING* l, int vocab_size, int batch_size)
{
    l->D = vocab_size;
    l->B = batch_size;
    l->h = allocmem(l->B,l->S,float);
    l->Wx = allocmem(l->D,l->E,float);
    typedef float (*ArrDE)[l->E];
    ArrDE Wx = (ArrDE) l->Wx;
    for (int i = 0; i < l->D; i++)
        for (int j = 0; j < l->E; j++)
            Wx[i][j] = urand(-0.5,0.5);
    if (l->padinx >= 0 && l->padinx < vocab_size)
        fltclr(Wx[l->padinx],l->E);
}

/* Frees the memory allocated by embedding_create() / embedding_init()
 * 
 * Parameters:
 *   l - Pointer to the neural network to be freed
 */
void embedding_free(EMBEDDING* l)
{
    freemem(l->h);
    freemem(l->Wx);
    freemem(l);
}

/* Resets the network hidden state.
 * 
 * Parameters:
 *   l - Pointer to the EMBEDDING neural network layer to be reset
 */
void embedding_reset(EMBEDDING* l)
{
    (void) l; /* Do nothing */
}
