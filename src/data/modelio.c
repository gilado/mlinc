/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store multi-layer neural network model */
#include <stdio.h>
#include "mem.h"
#include "array.h"
#include "arrayio.h"
#include "dense.h"
#include "denseio.h"
#include "lstm.h"
#include "lstmio.h"
#include "model.h"
#include "modelio.h"
#include "ctc.h"

/* read_model - Read a model from a file
 * 
 * Reads a model from the file pointed to by fp.
 * 
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 * 
 * Returns:
 *   Pointer to the read model if successful, NULL otherwise
 */
MODEL* read_model(FILE* fp)
{
    MODEL* m = allocmem(1,1,MODEL);
    int ok;
    int cnt = fscanf(fp," MODEL num_layers %d batch_size %d input_dim %d "
                     "add_bias %d output_dim %d loss_func '%c' optimizer '%c' "
                     "update_cnt %d normalize %d final %d\n",
                     &m->num_layers,&m->batch_size,&m->input_dim,
                     &m->add_bias,&m->output_dim,&m->loss_func,&m->optimizer,
                     &m->update_cnt,&m->normalize,&m->final);
    if (cnt < 10 || cnt == EOF) {
        fprintf(stderr,"In read_model: failed to read the header\n");
        goto err;
    }
    m->layer = allocmem(1,m->num_layers,LAYER);
    if (m->normalize) {
        int D = m->input_dim;           /* Input dimension: may include bias */
        int Dx = D - (1 - m->add_bias); /* Input dimension excluding bias    */
        m->mean = allocmem(1,Dx,float);
        m->sdev = allocmem(1,Dx,float);
        ok = read_array((fArr2D)m->mean,1,Dx,fp,0);
        if (ok)
            ok = read_array((fArr2D)m->sdev,1,Dx,fp,0);
        if (!ok) {
            fprintf(stderr,"In read_model: failed to read mean, sdev data\n");
            goto err;
        }
    }
    if (m->loss_func == 'C') { /* ctc */
        int T, L, blank;
        cnt = fscanf(fp," CTC T %d L %d blank %d\n",&T,&L,&blank);
        if (cnt < 3 || cnt == EOF) {
            fprintf(stderr,"In read_model: failed to read the ctc header\n");
            goto err;
        }
        m->ctc = ctc_create(T,L,blank);
    }
    for (int i = 0; i < m->num_layers; i++) {
        LAYER* l = &m->layer[i];
        cnt = fscanf(fp,
                     " LAYER type '%c' num_grads %d\n",&l->type,&l->num_grads);
        if (cnt < 2 || cnt == EOF) {
            fprintf(stderr,
                    "In read_model: failed to read layer %d header\n",i);
            goto err;
        }
        switch (l->type) {
            case 'd': 
                l->dense = read_dense(fp); 
                ok = (l->dense != NULL);
            break;
            case 'l': 
                l->lstm = read_lstm(fp); 
                ok = (l->lstm != NULL);
            break;
        }
        if (!ok) {
            fprintf(stderr,
                    "In read_model: failed to read layer %d data\n",i);
            goto err;
        }
        if (l->num_grads > 0) {
            /* grads is an array of pointers to arrays 
             * see model_compile() for layout
             */
            l->grads  = allocmem(1,l->num_grads,fArr2D*);
            ok = 1; /* assume success */
            switch (l->type) {
                case 'd': /* dense layer gradients */
                    for (int j = 0; j < l->num_grads && ok; j++) {
                        l->grads[j] = allocmem(l->dense->D,l->dense->S,float);
                        ok = read_array(l->grads[j],l->dense->D,l->dense->S,fp,0);
                    }
                break;
                case 'l': /* lstm layer gradients  */
                    for (int j = 0; j < l->num_grads && ok; j++) {
                        int rows = ((j / 4) % 2) ? l->lstm->S : l->lstm->D;
                        l->grads[j] = allocmem(rows,l->lstm->S,float);
                        ok = read_array(l->grads[j],rows,l->lstm->S,fp,0);
                    }
                break;
            }
            if (!ok) {
                fprintf(stderr,"In read_model: "
                        "failed to read layer %d gradient data\n",i);
                goto err;
            }
        }
    }
    return m;
err: /* error return */
    fflush(stderr);
    model_free(m);
    return NULL;
}

/* write_model - Write a model to a file
 * 
 * Writes the model pointed to by m to the file pointed to by fp. 
 * 
 * Parameters:
 *   m  - Pointer to the model to be written
 *   fp - Pointer to a FILE object representing the output file
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_model(const MODEL* m, FILE* fp)
{
    int ok;
    int cnt = fprintf(fp,"MODEL num_layers %d batch_size %d input_dim %d "
                 "add_bias %d output_dim %d loss_func '%c' optimizer '%c' " 
                 "update_cnt %d normalize %d final %d\n",
                 m->num_layers,m->batch_size,m->input_dim,
                 m->add_bias,m->output_dim,m->loss_func,m->optimizer,
                 m->update_cnt,m->normalize,m->final);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_model: failed to write the header\n");
        return 0;
    }
    if (m->normalize) {
        int D = m->input_dim;           /* Input dimension: may include bias */
        int Dx = D - (1 - m->add_bias); /* Input dimension excluding bias    */
        int ok = write_array((fArr2D)m->mean,1,Dx,fp,NULL,0);
        if (ok)
            ok = write_array((fArr2D)m->sdev,1,Dx,fp,NULL,0);
        if (!ok) {
            fprintf(stderr,"In write_model: failed to write mean, sdev data\n");
            return 0;
        }
    }
    if (m->ctc != NULL) {
        cnt = fprintf(fp,"CTC T %d L %d blank %d\n",
                                            m->ctc->T,m->ctc->L,m->ctc->blank);
        if (cnt <= 0 || cnt == EOF) {
            fprintf(stderr,"In write_model: failed to write the header\n");
            return 0;
        }
    }
    for (int i = 0; i < m->num_layers; i++) {
        LAYER* l = &m->layer[i];
        cnt = fprintf(fp,
                      "LAYER type '%c' num_grads %d\n",l->type,l->num_grads);
        if (cnt <= 0 || cnt == EOF) {
            fprintf(stderr,
                    "In write_model: failed to write layer %d header\n",i);
            return 0;
        }
        switch (l->type) {
            case 'd': ok = write_dense(l->dense,fp); break;
            case 'l': ok = write_lstm(l->lstm,fp); break;
        }
        if (!ok) {
            fprintf(stderr,
                    "In write_model: failed to write layer %d data\n",i);
            return 0;
        }
        if (l->num_grads > 0 && l->grads != NULL) {
            /* grads is an array of pointers to arrays 
             * see model_compile() for layout
             */
            ok = 1; /* assume success */
            switch (l->type) {
                case 'd': /* dense layer gradients */
                    for (int j = 0; j < l->num_grads && ok; j++) {
                        ok = write_array(l->grads[j],
                                         l->dense->D,l->dense->S,fp,NULL,0);
                    }
                break;
                case 'l': /* lstm layer gradients  */
                    for (int j = 0; j < l->num_grads && ok; j++) {
                        int rows = ((j / 4) % 2) ? l->lstm->S : l->lstm->D;
                        ok = write_array(l->grads[j],rows,l->lstm->S,fp,NULL,0);
                    }
                break;
            }
            if (!ok) {
                fprintf(stderr,"In write_model: "
                        "failed to write layer %d gradient data\n",i);
                return 0;
            }
        }
    }
    return 1;
}

/* load_model - Load a model from a file
 * 
 * Opens the file specified by the filename parameter for reading and 
 * loads a model from it.
 * 
 * Parameters:
 *   filename - Name of the file to load the model from
 * 
 * Returns:
 *   Pointer to the loaded model if successful, NULL otherwise
 */
MODEL* load_model(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_model: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    MODEL* m = read_model(fp);
    fclose(fp);
    return m;
}

/* store_model - Store a model into a file
 * 
 * Opens the file specified by the filename parameter for writing and 
 * stores the model pointed to by m into it.
 * 
 * Parameters:
 *   m        - Pointer to the model to be stored
 *   filename - Name of the file to store the model in
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_model(const MODEL* m, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_model: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_model(m,fp);
    fclose(fp);
    return ok;
}

