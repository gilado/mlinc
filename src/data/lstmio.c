/* Copyright (c) 2023-2024 Gilad Odinak */
/* Functions to load and store NN lstm layer */
#include <stdio.h>
#include "mem.h"
#include "array.h"
#include "arrayio.h"
#include "lstm.h"
#include "lstmio.h"

/* read_lstm - Read an LSTM layer from a file
 * 
 * Reads an LSTM layer from the file pointed to by fp.
 * 
 * Parameters:
 *   fp - Pointer to a FILE object representing the input file
 * 
 * Returns:
 *   Pointer to the read LSTM layer if successful, NULL otherwise
 */
LSTM* read_lstm(FILE* fp)
{
    int D, S, B, b;
    char c;
    int cnt = fscanf(fp," LSTM D %d S %d B %d "
                        "activation '%c' stateful %d\n",&D,&S,&B,&c,&b);
    if (cnt < 5 || cnt == EOF) {
        fprintf(stderr,"In read_lstm: failed to read header\n");
        return NULL;
    }
    if (c != 'n' && c != 'r' && c != 's' && c != 'S') {
        fprintf(stderr,"In read_lstm: invalid activation code\n");
        return NULL;
    }
    LSTM* l = allocmem(1,1,LSTM);
    l->S = S;
    l->D = D;
    l->B = B;
    l->activation = c;
    l->stateful = (b) ? 1 : 0;

    l->f = allocmem(l->B,l->S,float);
    l->i = allocmem(l->B,l->S,float);
    l->o = allocmem(l->B,l->S,float);
    l->cc = allocmem(l->B+1,l->S,float);
    l->h = allocmem(l->B+1,l->S,float);
    l->c = allocmem(l->B+1,l->S,float);
    l->Wf = allocmem(l->D,l->S,float);
    l->Wi = allocmem(l->D,l->S,float);
    l->Wc = allocmem(l->D,l->S,float);
    l->Wo = allocmem(l->D,l->S,float);
    l->Uf = allocmem(l->S,l->S,float);
    l->Ui = allocmem(l->S,l->S,float);
    l->Uc = allocmem(l->S,l->S,float);
    l->Uo = allocmem(l->S,l->S,float);
    l->ph = allocmem(1,l->S,float);
    l->pc = allocmem(1,l->S,float);

    fArr2D Wx[4] = {l->Wf,l->Wi,l->Wc,l->Wo};
    char* sWx[4] = {"Wf","Wi","Wc","Wo"};
    for (int i = 0; i < 4; i++) {
        int ok = read_array(Wx[i],l->D,l->S,fp,0);
        if (!ok) {
            fprintf(stderr,"In read_lstm: failed to read %s weights\n",sWx[i]);
            goto err;
        }
    }
    fArr2D Ux[4] = {l->Uf,l->Ui,l->Uc,l->Uo};
    char* sUx[4] = {"Uf","Ui","Uc","Uo"};
    for (int i = 0; i < 4; i++) {
        int ok = read_array(Ux[i],l->S,l->S,fp,0);
        if (!ok) {
            fprintf(stderr,"In read_lstm: failed to read %s weights\n",sUx[i]);
            goto err;
        }
    }
    fVec px[2] = {l->ph,l->pc};
    char* spx[2] = {"hidden","cell"};
    for (int i = 0; i < 2; i++) {
        int ok = read_array((fArr2D)px[i],1,l->S,fp,0);
        if (!ok) {
            fprintf(stderr,"In read_lstm: failed to read %s state\n",spx[i]);
            return 0;
        }
    }
    return l;
        
err: /* error exit */
    freemem(l->f);
    freemem(l->i);
    freemem(l->o);
    freemem(l->cc);
    freemem(l->h);
    freemem(l->c);
    freemem(l->Wf);
    freemem(l->Wi);
    freemem(l->Wc);
    freemem(l->Wo);
    freemem(l->Uf);
    freemem(l->Ui);
    freemem(l->Uc);
    freemem(l->Uo);
    freemem(l->ph);
    freemem(l->pc);
    freemem(l);
    return NULL;
}

/* write_lstm - Write an LSTM layer to a file
 * 
 * Writes the LSTM layer pointed to by d to the file pointed to by fp. 
 * 
 * Parameters:
 *   d  - Pointer to the LSTM layer to be written
 *   fp - Pointer to a FILE object representing the output file
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int write_lstm(const LSTM* l, FILE* fp)
{
    int cnt = fprintf(fp,"LSTM D %d S %d B %d activation '%c' stateful %d\n",
                                     l->D,l->S,l->B,l->activation,l->stateful);
    if (cnt <= 0 || cnt == EOF) {
        fprintf(stderr,"In write_lstm: failed to write the header\n");
        return 0;
    }

    fArr2D Wx[4] = {l->Wf,l->Wi,l->Wc,l->Wo};
    char* sWx[4] = {"Wf","Wi","Wc","Wo"};
    for (int i = 0; i < 4; i++) {
        int ok = write_array(Wx[i],l->D,l->S,fp,NULL,0);
        if (!ok) {
            fprintf(stderr,
                    "In write_lstm: failed to write %s weights\n",sWx[i]);
            return 0;
        }
    }
    fArr2D Ux[4] = {l->Uf,l->Ui,l->Uc,l->Uo};
    char* sUx[4] = {"Uf","Ui","Uc","Uo"};
    for (int i = 0; i < 4; i++) {
        int ok = write_array(Ux[i],l->S,l->S,fp,NULL,0);
        if (!ok) {
            fprintf(stderr,
                    "In write_lstm: failed to write %s weights\n",sUx[i]);
            return 0;
        }
    }
    fVec px[2] = {l->ph,l->pc};
    char* spx[2] = {"hidden","cell"};
    for (int i = 0; i < 2; i++) {
        int ok = write_array((fArr2D)px[i],1,l->S,fp,NULL,0);
        if (!ok) {
            fprintf(stderr,"In write_lstm: failed to write %s state\n",spx[i]);
            return 0;
        }
    }
    return 1;
}

/* load_lstm - Load an LSTM layer from a file
 * 
 * Opens the file specified by the filename parameter for reading and 
 * loads an LSTM layer from it.
 * 
 * Parameters:
 *   filename - Name of the file to load the LSTM layer from
 * 
 * Returns:
 *   Pointer to the loaded LSTM layer if successful, NULL otherwise
 */
LSTM* load_lstm(const char* filename)
{
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"In load_lstm: failed to open file '%s' for read\n",filename);
        return NULL;
    }
    LSTM* l = read_lstm(fp);
    fclose(fp);
    return l;
}

/* store_lstm - Store an LSTM layer into a file
 * 
 * Opens the file specified by the filename parameter for writing and 
 * stores the LSTM layer pointed to by d into it.
 * 
 * Parameters:
 *   d        - Pointer to the LSTM layer to be stored
 *   filename - Name of the file to store the LSTM layer in
 * 
 * Returns:
 *   1 if successful, 0 otherwise
 */
int store_lstm(const LSTM* l, const char* filename)
{
    FILE* fp = fopen(filename,"wb");
    if (fp == NULL) {
        fprintf(stderr,"In store_lstm: failed to open file '%s' for write\n",filename);
        return 0;
    }
    int ok = write_lstm(l,fp);
    fclose(fp);
    return ok;
}

