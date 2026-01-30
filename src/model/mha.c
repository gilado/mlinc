/* Copyright (c) 2026 Gilad Odinak */
/* Multi-Head Attention layer functions */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include "mem.h"
#include "mha.h"

MHA* mha_create(int heads, int steps)
{
    MHA* l = allocmem(1,1,MHA);
    l->H = heads;
    l->T = steps;
    return l;
}

void mha_init(MHA* l, int input_dim, int batch_size, int training)
{
    if (input_dim % l->H != 0) {
        fflush(stdout);
        fprintf(stderr,"mha_init: input_dim %d not an integral multiple of heads %d\n",input_dim,l->H);
        freemem(l);
        exit(-1);
    }
    l->D = input_dim;
    l->Dh = input_dim / l->H;   
    l->B = batch_size;
    l->BT = l->B * l->T;
    
    l->Wq = allocmem(l->D,l->D,float);
    l->Wk = allocmem(l->D,l->D,float);
    l->Wv = allocmem(l->D,l->D,float);
    l->Wo = allocmem(l->D,l->D,float);

    l->Q = allocmem(l->BT,l->D,float);
    l->K = allocmem(l->BT,l->D,float);
    l->V = allocmem(l->BT,l->D,float);

    l->Qh = allocmem(l->T,l->Dh,float);
    l->Kh = allocmem(l->T,l->Dh,float);
    l->Vh = allocmem(l->T,l->Dh,float);

    l->Scores = allocmem(l->T,l->T,float);
    l->Att = allocmem(l->T,l->T,float);
    l->Oh = allocmem(l->T,l->Dh,float);
    
    l->Out = allocmem(l->BT,l->D,float);

    if (!training)
        return;

    l->dOut = allocmem(l->BT,l->D,float);

    l->dQ = allocmem(l->BT,l->D,float);
    l->dK = allocmem(l->BT,l->D,float);
    l->dV = allocmem(l->BT,l->D,float);

    l->dQh = allocmem(l->T,l->Dh,float);
    l->dKh = allocmem(l->T,l->Dh,float);
    l->dVh = allocmem(l->T,l->Dh,float);

    l->dOh = allocmem(l->T,l->Dh,float);
    l->dAtt = allocmem(l->T,l->T,float);
    l->dScores = allocmem(l->T,l->T,float);

    l->gWq = allocmem(l->D,l->D,float);
    l->gWk = allocmem(l->D,l->D,float);
    l->gWv = allocmem(l->D,l->D,float);
    l->gWo = allocmem(l->D,l->D,float);    
}

void mha_free(MHA* l)
{
    freemem(l->Wq);
    freemem(l->Wk);
    freemem(l->Wv);
    freemem(l->Wo);

    freemem(l->Q);
    freemem(l->K);
    freemem(l->V);

    freemem(l->Qh);
    freemem(l->Kh);
    freemem(l->Vh);

    freemem(l->Scores);
    freemem(l->Att);
    freemem(l->Oh);
    
    freemem(l->Out);

    /* backward buffers */
    freemem(l->dOut);

    freemem(l->dQ);
    freemem(l->dK);
    freemem(l->dV);

    freemem(l->dQh);
    freemem(l->dKh);
    freemem(l->dVh);

    freemem(l->dOh);
    freemem(l->dAtt);
    freemem(l->dScores);

    freemem(l->gWq);
    freemem(l->gWk);
    freemem(l->gWv);
    freemem(l->gWo);

    freemem(l);
}
