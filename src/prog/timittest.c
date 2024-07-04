/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include "mem.h"
#include "float.h"
#include "array.h"
#include "model.h"
#include "modelio.h"
#include "featfile.h"
#include "beamsrch.h"
#include "alignseq.h"

/* Each sequence contains multiple phonemes, find how many in total */
int count_phoneme(int*  yc, int len)
{
    int cnt = 0;
    for (int i = 0; i < len; i++) {
        if (yc[i] >= EOP) {
            yc[i] -= EOP;
            cnt++;
        }
    }
    return cnt;
}


/* Removes consecutive duplicate labels and blanks from the labels array.
 * Modifies the array labels in place and returns the new length.
 */
int dedup_labels(iVec restrict labels, int len, int blank_inx)
{
    int j, k;
    for (j = 0, k = 0; j < len; j++)
        if (labels[j] != blank_inx)
            if (k == 0 || labels[k - 1] != labels[j])
                labels[k++] = labels[j];
    return k;
}


int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,"syntax: timittest <model file> <feature file>\n");
        return -1;
    }        
    const char* modelfile = argv[1];
    const char* featfile = argv[2];

    MODEL* m = load_model(modelfile);
    if (m == NULL) {
        fprintf(stderr,"Failed to load model file '%s'\n",modelfile);
        return -1;
    }

    int D = m->input_dim;   /* Input vectors dimension          */
    int N = m->output_dim;  /* Number of output classes         */
    int MTe = 1000;         /* Max number of samples 10 seconds */
    float xTe[MTe][D];      /* Input feature vectors            */
    int   yTec[MTe];        /* Phoneme indices (true labels)    */

    FILE* fp = fopen(featfile,"rb");
    if (fp == NULL) {
        fprintf(stderr,"Failed to open file '%s' for read\n",featfile);
        model_free(m);
        return -1;    
    }
    int cnt = read_feature_file(fp,MTe,xTe,yTec);
    fclose(fp);
    if (cnt == 0) {
        printf("feature file does not contain any data\n");
        model_free(m);
        return 0;
    }
    MTe = cnt; /* Actual number of samples */
    /* Each sequence contains multiple phonemes, find how many in total */
    int PTe = count_phoneme(yTec,MTe);

    printf("%d phonemes, %d samples\n\n",PTe,MTe);

    float yp[MTe][N];   /* Predicted probabilities */
    int ypc[MTe];       /* Predicted labels        */
    int ytc[MTe];       /* True labels             */

    /* Convert true labels to phonemes: merge repeated labels, remove silence */
    memcpy(ytc,yTec,MTe * sizeof(int));
    int ytc_len = dedup_labels(ytc,MTe,SIL);

    /* Predict lables from features */    
    model_predict(m,xTe,yp,MTe);

    /* Performs beam search to find the most probable sequences of lables */
    int beamwidth = 3;
    int timesteps = MTe;
    int sequences[beamwidth][timesteps + 1];
    float scores[beamwidth];

    int nc = REDUCED_PHONEME_CNT;
    beam_search(yp,timesteps,nc,beamwidth,sequences,scores);
    int ypc_len = dedup_labels(sequences[0],timesteps,SIL);
    memcpy(ypc,sequences[0],ypc_len * sizeof(int));

    /* Align the true and predicted phoneme sequences */
    int rlen = ((ytc_len > ypc_len) ? ytc_len : ypc_len) * 2;
    int ypc2[rlen];
    int ytc2[rlen];
    alignseq(ypc,ypc_len,ytc,ytc_len,ypc2,ytc2,rlen,SIL);

    /* Print the sequences */
    printf("True phonemes:      ");
    for (int i = 0; i < rlen; i++) {
        if (ytc2[i] != SIL || ypc2[i] != SIL)
            printf("%-3s ",reduced_phoneme_names[ytc2[i]]);
        else
            break;
    }
    printf("\n");
    printf("Predicted phonemes: ");
    for (int i = 0; i < rlen; i++) {
        if (ytc2[i] != SIL || ypc2[i] != SIL)
            printf("%-3s ",reduced_phoneme_names[ypc2[i]]);
        else
            break;
    }
    printf("\n");

    /* Count matches */
    int pcnt = 0, mcnt = 0;
    for (int i = 0; i < rlen; i++) {
        if (ytc2[i] != SIL || ypc2[i] != SIL) {
            pcnt++;
            if (ytc2[i] == ypc2[i])
                mcnt++;
        }
        else
            break;
    }
    printf("%d out of %d matched (%5.3f)\n",mcnt,pcnt,(float) mcnt / pcnt);

    model_free(m);
}
