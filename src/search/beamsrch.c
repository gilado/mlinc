/* Copyright (c) 2023-2024 Gilad Odinak */
/* Beam search decoder             */
#include <math.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "beamsrch.h"

struct can_seq_s {
    int* seq;
    float score;
};

static inline int can_seq_cmp(const void* a, const void* b)
{
    float as = ((const struct can_seq_s*) a)->score;
    float bs = ((const struct can_seq_s*) b)->score;
    return (as > bs) - (as < bs);
}

void beam_search(fArr2D probabilities_,
                 int T, int C, int beam_width,
                 iArr2D sequences_, fVec scores_)
{
    const int B = beam_width;
    typedef float (*fArrTC)[C];
    typedef int (*iArrBT1)[T + 1];
    typedef float (*fVecB);
    fArrTC probabilities = (fArrTC) probabilities_;
    iArrBT1 sequences = (iArrBT1) sequences_;
    fVecB scores = (fVecB) scores_;

    /* Use heap for really large sequences, stack otherwise */
    int useheap = (B * C) * ( T + 1) > 65536;
    float ns[(useheap) ? 1 : (B * C)][(useheap) ? 1 : (T + 1)];
    iArrBT1 new_seqs = (iArrBT1) ((useheap) ? allocmem(B * C,T + 1,float) : ns);

    struct can_seq_s can_seqs[B * C];

    for (int i = 0; i < B; i++)
        for (int j = 0; j <= T; j++)
            sequences[i][j] = 0;
    scores[0] = 0.0;
    int num_sequences = 1;

    for (int t = 0; t < T; ++t) {
        int num_can = 0;
        for (int i = 0; i < num_sequences; i++) {
            for (int c = 0; c < C; c++) {
                int* new_seq = new_seqs[num_can];
                memcpy(new_seq,sequences[i],t * sizeof(int));
                new_seq[t] = c;
                float new_score = scores[i] - log(probabilities[t][c]);

                can_seqs[num_can].seq = new_seq;
                can_seqs[num_can].score = new_score;
                num_can++;
            }
        }

        qsort(can_seqs,num_can,sizeof(struct can_seq_s),can_seq_cmp);

        for (int i = 0; i < beam_width; i++) {
            memcpy(sequences[i],can_seqs[i].seq,(t + 1) * sizeof(int));
            scores[i] = can_seqs[i].score;
        }
        num_sequences = beam_width;
    }
    if (useheap)
        freemem(new_seqs);
}
