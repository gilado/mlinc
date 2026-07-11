/* Copyright (c) 2023-2024 Gilad Odinak */
/* Model layer abstraction implementation.                              */
/* All per-layer-type dispatch lives here so that model.c can call the  */
/* layer_*() wrappers uniformly.                                        */
#include <stdio.h>
#include <stdlib.h>
#include "mem.h"
#include "array.h"
#include "clip.h"
#include "adamw.h"
#include "dense.h"
#include "lstm.h"
#include "transformer.h"
#include "layer.h"

/* Updates all weights in array w[M][N], according to the corresponding
 * gradients in g[M][N], using linear update.
 * The rate of update is controlled by learning_rate, weight_decay.
 */
static inline void linear_update(fArr2D w_/*[M][N]*/, fArr2D g_/*[M][N]*/,
                                 int M, int N,
                                 float learning_rate, float weight_decay)
{
    typedef float (*ArrMN)[N];
    ArrMN w = (ArrMN) w_;
    ArrMN g = (ArrMN) g_;

    clip_gradients(g,M,N,1.0e-12,10.0);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            w[i][j] -= learning_rate * (g[i][j] + weight_decay * w[i][j]);
}

int layer_init(LAYER* l, int input_dim, int batch_size)
{
    switch (l->type) {
        case 'd':
            dense_init(l->dense,input_dim,batch_size);
            return l->dense->S;
        case 'l':
            lstm_init(l->lstm,input_dim,batch_size);
            return l->lstm->S;
        case 't': {
            /* The transformer's model dimension D and sequence length T are
             * fixed at transformer_create(). input_dim must equal D, and the
             * model's row count (batch_size) must be B*T for whole sequences
             * of length T, so the number of sequences is batch_size / T. */
            TRANSFORMER* tr = l->transformer;
            if (input_dim != tr->D) {
                fflush(stdout);
                fprintf(stderr,"layer_init: transformer input_dim %d "
                        "!= model_dim %d\n",input_dim,tr->D);
                exit(-1);
            }
            if (tr->T <= 0 || batch_size % tr->T != 0) {
                fflush(stdout);
                fprintf(stderr,"layer_init: batch_size %d not a multiple "
                        "of transformer T %d\n",batch_size,tr->T);
                exit(-1);
            }
            transformer_init(tr,batch_size / tr->T,1,0.0);
            l->out = allocmem(tr->BT,tr->D,float);
            return tr->D;
        }
    }
    layer_unsupported("layer_init",l->type);
    return 0; /* not reached */
}

void layer_reset(LAYER* l)
{
    switch (l->type) {
        case 'd': dense_reset(l->dense); break;
        case 'l': lstm_reset(l->lstm); break;
        case 't': /* transformer carries no cross-batch state */ break;
    }
}

void layer_free(LAYER* l)
{
    switch (l->type) {
        case 'd': dense_free(l->dense); break;
        case 'l': lstm_free(l->lstm); break;
        case 't': transformer_free(l->transformer); freemem(l->out); break;
    }
}

void layer_set_batch_size(LAYER* l, int batch_size)
{
    switch (l->type) {
        case 'd': dense_set_batch_size(l->dense,batch_size); break;
        case 'l': lstm_set_batch_size(l->lstm,batch_size); break;
        case 't':
            fflush(stdout);
            fprintf(stderr,
                "layer_set_batch_size: not supprted by transformer layer\n");
            exit(-1);
        break;
    }
}

void layer_alloc_grads(LAYER* l, char optimizer)
{
    switch (l->type) {
        case 'd': {
            int D = l->dense->D;
            int S = l->dense->S;
            int ng = 0;             /* Number of gradient related arrays */
            switch (optimizer) {
                case 'l': ng = 1; break; /* gWx[D][S]                    */
                case 'a': ng = 3; break; /* gWx mWx vWx  [D][S]          */
            }
            fArr2D* g = allocmem(1,ng,fArr2D*);
            for (int j = 0; j < ng; j++)
                g[j] = allocmem(D,S,float);
            l->grads = g;
            l->num_grads = ng;
        }
        break;
        case 'l': {
            int D = l->lstm->D;
            int S = l->lstm->S;
            int ng = 0; /* Number of gradient related arrays   */
            /* gW{f,i,c,o}[D][S] gU{f,i,c,o}[S][S]             */
            switch (optimizer) {
                case 'l': ng = 8; break;                                    
                case 'a': ng = 24; break; /* linear + adam m/v */
            }
            fArr2D* g = allocmem(1,ng,fArr2D*);
            for (int j = 0; j < ng; j++)
                g[j] = allocmem(((j / 4) % 2) ? S : D,S,float);
            l->grads = g;
            l->num_grads = ng;
        }
        break;
        case 't': {
            /* The transformer owns its 10 weight gradients internally
             * (mha->gW{q,k,v,o}, gWx1, gWx2, dg1/db1, dg2/db2), so the
             * linear optimizer needs no extra buffers. AdamW additionally
             * needs per-parameter m/v moments, which are NOT owned by the
             * transformer and are allocated here, shaped to match each
             * parameter:
             *   [0..3] Wq,Wk,Wv,Wo   [D][D]
             *   [4]    ffn1->Wx       [D][Dff]
             *   [5]    ffn2->Wx       [Dff][D]
             *   [6..9] norm gamma/beta [D][1]
             * Layout: g[0..9] = m, g[10..19] = v (gradients stay internal). */
            TRANSFORMER* tr = l->transformer;
            int D = tr->D;
            int Dff = tr->Dff;
            if (optimizer == 'l') {
                l->grads = NULL;
                l->num_grads = 0;
            } else { /* 'a' adamw */
                int rows[10] = { D, D, D, D, D,   Dff, D, D, D, D };
                int cols[10] = { D, D, D, D, Dff, D,   1, 1, 1, 1 };
                int ng = 20;
                fArr2D* g = allocmem(1,ng,fArr2D*);
                for (int j = 0; j < 10; j++) {
                    g[j]      = allocmem(rows[j],cols[j],float); /* m */
                    g[j + 10] = allocmem(rows[j],cols[j],float); /* v */
                }
                l->grads = g;
                l->num_grads = ng;
            }
        }
        break;
    }
}

void layer_update(LAYER* l, char optimizer,
                  float learning_rate, float weight_decay, int update_cnt)
{
    float lr = learning_rate;
    float wd = weight_decay;
    int uc = update_cnt;
    fArr2D* g = l->grads;
    switch (l->type) {
        case 'd': { /* dense */
            DENSE* ld = l->dense;
            int D = ld->D;
            int S = ld->S;
            switch (optimizer) {
                case 'l': /* linear */
                    linear_update(ld->Wx,g[0],D,S,lr,wd);
                break;
                case 'a': /* adamw */
                    adamw_update(ld->Wx,g[0],g[1],g[2],D,S,lr,wd,uc);
                break;
            }
        }
        break;
        case 'l': { /* lstm */
            LSTM* ll = l->lstm;
            int D = ll->D;
            int S = ll->S;
            switch (optimizer) {
                case 'l': /* linear */
                    linear_update(ll->Wf,g[0],D,S,lr,wd);
                    linear_update(ll->Wi,g[1],D,S,lr,wd);
                    linear_update(ll->Wc,g[2],D,S,lr,wd);
                    linear_update(ll->Wo,g[3],D,S,lr,wd);
                    linear_update(ll->Uf,g[4],S,S,lr,wd);
                    linear_update(ll->Ui,g[5],S,S,lr,wd);
                    linear_update(ll->Uc,g[6],S,S,lr,wd);
                    linear_update(ll->Uo,g[7],S,S,lr,wd);
                break;
                case 'a': /* adamw */
                    adamw_update(ll->Wf,g[0],g[0+8],g[0+16],D,S,lr,wd,uc);
                    adamw_update(ll->Wi,g[1],g[1+8],g[1+16],D,S,lr,wd,uc);
                    adamw_update(ll->Wc,g[2],g[2+8],g[2+16],D,S,lr,wd,uc);
                    adamw_update(ll->Wo,g[3],g[3+8],g[3+16],D,S,lr,wd,uc);
                    adamw_update(ll->Uf,g[4],g[4+8],g[4+16],S,S,lr,wd,uc);
                    adamw_update(ll->Ui,g[5],g[5+8],g[5+16],S,S,lr,wd,uc);
                    adamw_update(ll->Uc,g[6],g[6+8],g[6+16],S,S,lr,wd,uc);
                    adamw_update(ll->Uo,g[7],g[7+8],g[7+16],S,S,lr,wd,uc);
                break;
            }
        }
        break;
        case 't': { /* transformer */
            TRANSFORMER* tr = l->transformer;
            MHA* mha = tr->mha;
            int D = tr->D;
            int Dff = tr->Dff;
            /* Gradients are read straight from the transformer's internal
             * buffers. For AdamW the moments live in g[0..9] (m) and
             * g[10..19] (v), in the same parameter order as the updates
             * below. Note: weight decay is applied uniformly, including to
             * the norm gamma/beta; set weight_decay to 0 to disable. */
            switch (optimizer) {
                case 'l': /* linear */
                    linear_update(mha->Wq,mha->gWq,D,D,lr,wd);
                    linear_update(mha->Wk,mha->gWk,D,D,lr,wd);
                    linear_update(mha->Wv,mha->gWv,D,D,lr,wd);
                    linear_update(mha->Wo,mha->gWo,D,D,lr,wd);
                    linear_update(tr->ffn1->Wx,tr->gWx1,D,Dff,lr,wd);
                    linear_update(tr->ffn2->Wx,tr->gWx2,Dff,D,lr,wd);
                    linear_update((fArr2D) tr->norm1->gamma,(fArr2D) tr->dg1,D,1,lr,wd);
                    linear_update((fArr2D) tr->norm1->beta,(fArr2D) tr->db1,D,1,lr,wd);
                    linear_update((fArr2D) tr->norm2->gamma,(fArr2D) tr->dg2,D,1,lr,wd);
                    linear_update((fArr2D) tr->norm2->beta,(fArr2D) tr->db2,D,1,lr,wd);
                break;
                case 'a': /* adamw */
                    adamw_update(mha->Wq,mha->gWq,g[0],g[10],D,D,lr,wd,uc);
                    adamw_update(mha->Wk,mha->gWk,g[1],g[11],D,D,lr,wd,uc);
                    adamw_update(mha->Wv,mha->gWv,g[2],g[12],D,D,lr,wd,uc);
                    adamw_update(mha->Wo,mha->gWo,g[3],g[13],D,D,lr,wd,uc);
                    adamw_update(tr->ffn1->Wx,tr->gWx1,g[4],g[14],D,Dff,lr,wd,uc);
                    adamw_update(tr->ffn2->Wx,tr->gWx2,g[5],g[15],Dff,D,lr,wd,uc);
                    adamw_update((fArr2D) tr->norm1->gamma,(fArr2D) tr->dg1,g[6],g[16],D,1,lr,wd,uc);
                    adamw_update((fArr2D) tr->norm1->beta,(fArr2D) tr->db1,g[7],g[17],D,1,lr,wd,uc);
                    adamw_update((fArr2D) tr->norm2->gamma,(fArr2D) tr->dg2,g[8],g[18],D,1,lr,wd,uc);
                    adamw_update((fArr2D) tr->norm2->beta,(fArr2D) tr->db2,g[9],g[19],D,1,lr,wd,uc);
                break;
            }
        }
        break;
    }
}
