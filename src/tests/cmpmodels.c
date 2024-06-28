/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to compare models and their components */
#include <stdio.h>
#include "array.h"
#include "dense.h"
#include "lstm.h"
#include "model.h"

int compare_arrays(fArr2D a1_, fArr2D a2_, int rows, int cols)
{
    typedef float (*ArrMN)[cols];
    ArrMN a1 = (ArrMN) a1_;
    ArrMN a2 = (ArrMN) a2_;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (fabsf(a1[i][j] - a2[i][j]) > 1e-6)
                return 0;
    return 1;
}
                
void compare_dense(DENSE* l1, DENSE* l2, 
                   fArr2D* grads1, fArr2D* grads2, int num_grads, int lyr)
{
    if (l1->D != l2->D)
        printf("layer %d l1->D %d l2->D %d\n",lyr,l1->D,l2->D);
    if (l1->S != l2->S)
        printf("layer %d l1->S %d l2->S %d\n",lyr,l1->S,l2->S);
    if (l1->B != l2->B)
        printf("layer %d l1->B %d l2->B %d\n",lyr,l1->B,l2->B);
    if (l1->activation != l2->activation)
        printf("layer %d l1->activation '%c' l2->activation '%c'\n",lyr,l1->activation,l2->activation);
    if (l1->D != l2->D || l1->S != l2->S || l1->B != l2->B || l1->activation != l2->activation)
        exit(-1);
    if (!compare_arrays(l1->h,l2->h,l1->B,l1->S))
        printf("layer %d h arrays differ\n",lyr);
    if (!compare_arrays(l1->Wx,l2->Wx,l1->D,l1->S))
        printf("layer %d Wx arrays differ\n",lyr);
    for (int i = 0; i < num_grads; i++)
        if (!compare_arrays(grads1[i],grads2[i],l1->D,l1->S))
            printf("layer %d grads[%d] arrays differ\n",lyr,i);  
}

void compare_lstm(LSTM* l1, LSTM* l2, 
                   fArr2D* grads1, fArr2D* grads2, int num_grads, int lyr)
{
    if (l1->D != l2->D)
        printf("layer %d l1->D %d l2->D %d\n",lyr,l1->D,l2->D);
    if (l1->S != l2->S)
        printf("layer %d l1->S %d l2->S %d\n",lyr,l1->S,l2->S);
    if (l1->B != l2->B)
        printf("layer %d l1->B %d l2->B %d\n",lyr,l1->B,l2->B);
    if (l1->activation != l2->activation)
        printf("layer %d l1->activation '%c' l2->activation '%c'\n",lyr,l1->activation,l2->activation);
    if (l1->stateful != l2->stateful)
        printf("layer %d l1->stateful %d l2->stateful %d\n",lyr,l1->stateful,l2->stateful);
    if (l1->D != l2->D || l1->S != l2->S || l1->B != l2->B || l1->activation != l2->activation || l1->stateful != l2->stateful)
        exit(-1);
    if (!compare_arrays(l1->Wf,l2->Wf,l1->D,l1->S))
        printf("layer %d Wf arrays differ\n",lyr);
    if (!compare_arrays(l1->Wi,l2->Wi,l1->D,l1->S))
        printf("layer %d Wi arrays differ\n",lyr);
    if (!compare_arrays(l1->Wc,l2->Wc,l1->D,l1->S))
        printf("layer %d Wc arrays differ\n",lyr);
    if (!compare_arrays(l1->Wo,l2->Wo,l1->D,l1->S))
        printf("layer %d Wo arrays differ\n",lyr);
    if (!compare_arrays(l1->Uf,l2->Uf,l1->S,l1->S))
        printf("layer %d Uf arrays differ\n",lyr);
    if (!compare_arrays(l1->Ui,l2->Ui,l1->S,l1->S))
        printf("layer %d Ui arrays differ\n",lyr);
    if (!compare_arrays(l1->Uc,l2->Uc,l1->S,l1->S))
        printf("layer %d Uc arrays differ\n",lyr);
    if (!compare_arrays(l1->Uo,l2->Uo,l1->S,l1->S))
        printf("layer %d Uo arrays differ\n",lyr);
    if (!compare_arrays((fArr2D)l1->pc,(fArr2D)l2->pc,1,l1->S))
        printf("layer %d pc arrays differ\n",lyr);
    if (!compare_arrays((fArr2D)l1->ph,(fArr2D)l2->ph,1,l1->S))
        printf("layer %d ph arrays differ\n",lyr);
    for (int i = 0; i < num_grads; i++) {
        int R = ((i / 4) % 2) ? l1->S : l1->D;
        if (!compare_arrays(grads1[i],grads2[i],R,l1->S))
            printf("layer %d grads[%d] arrays differ\n",lyr,i);
    }
}

void compare_layers(LAYER* l1, LAYER* l2, int lyr)
{
    if (l1->type != l2->type)
        printf("m1->layer[%d] type '%c' m2->ilayer[%d] type '%c'\n",lyr,l1->type,lyr,l2->type);
    if (l1->num_grads != l2->num_grads)
        printf("m1->layer[%d] num_grads %d m2->ilayer[%d] num_grads %d\n",lyr,l1->num_grads,lyr,l2->num_grads);
    if (l1->type != l2->type || l1->num_grads != l2->num_grads)
        exit(-1);
    switch(l1->type) {
        case 'd':
            compare_dense(l1->dense,l2->dense,l1->grads,l2->grads,l1->num_grads,lyr);
        break;
        case 'l':
            compare_lstm(l1->lstm,l2->lstm,l1->grads,l2->grads,l1->num_grads,lyr);
        break;
        default:
            printf("unknown layer type '%c'\n",l1->type);
            exit(-1);
    }
}

void compare_models(MODEL* m1, MODEL* m2)
{
    if (m1->num_layers != m2->num_layers)
        printf("m1->num_layers %d m2->num_layers %d\n",m1->num_layers,m2->num_layers);
    if (m1->batch_size != m2->batch_size)
        printf("m1->batch_size %d m2->batch_size %d\n",m1->batch_size,m2->batch_size);
    if (m1->input_dim != m2->input_dim)
        printf("m1->input_dim %d m2->input_dim %d\n",m1->input_dim,m2->input_dim);
    if (m1->add_bias != m2->add_bias)
        printf("m1->add_bias %d m2->add_bias %d\n",m1->add_bias,m2->add_bias);
    if (m1->output_dim != m2->output_dim)
        printf("m1->output_dim %d m2->output_dim %d\n",m1->output_dim,m2->output_dim);
    if (m1->loss_func != m2->loss_func)
        printf("m1->loss_func '%c' m2->loss_func '%c'\n",m1->loss_func,m2->loss_func);
    if (m1->ctc != m2->ctc)
        printf("m1->ctc %p m2->ctc %p\n",m1->ctc,m2->ctc);
    if (m1->optimizer != m2->optimizer)
        printf("m1->optimizer '%c' m2->optimizer '%c'\n",m1->optimizer,m2->optimizer);
    if (m1->update_cnt != m2->update_cnt)
        printf("m1->update_cnt %d m2->update_cnt %d\n",m1->update_cnt,m2->update_cnt);
    if (m1->normalize != m2->normalize)
        printf("m1->normalize %d m2->normalize %d\n",m1->normalize,m2->normalize);
    if (m1->final != m2->final)
        printf("m1->final %d m2->final %d\n",m1->final,m2->final);
    if (m1->num_layers != m2->num_layers)
        exit(-1);
    for (int i = 0; i < m1->num_layers; i++)
        compare_layers(&m1->layer[i],&m2->layer[i],i);
}
