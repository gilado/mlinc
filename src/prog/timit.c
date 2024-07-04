/* Copyright (c) 2023-2024 Gilad Odinak */
/* Trains a multi layer LSTM followed by Dense layer to recognize spoken
 * phonemse from the TIMIT dataset.
 */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>
#include <math.h>
#include "mem.h"
#include "float.h"
#include "etime.h"
#include "random.h"
#include "array.h"
#include "dense.h"
#include "lstm.h"
#include "accuracy.h"
#include "model.h"
#include "modelio.h"
#include "featfile.h"
#include "onehot.h"
#include "editdist.h"
#include "beamsrch.h"
#include "alignseq.h"

/* Directoires of phoneme feature files (no trailing slash) */
const char* timit_tr_data_dir = "data/timit/features/train";
const char* timit_vd_data_dir = "data/timit/features/validate";
const char* timit_te_data_dir = "data/timit/features/test";

/* Lists of training, validation and test feature files */
const char* timit_tr_file_list = "data/timit/tr_file.lst";
const char* timit_vd_file_list = "data/timit/vd_file.lst";
const char* timit_te_file_list = "data/timit/te_file.lst";

#define TIMIT_TR_MAX_SEQUENCE_CNT    5000
#define TIMIT_TR_MAX_SAMPLE_CNT   1500000
#define TIMIT_VD_MAX_SEQUENCE_CNT     500
#define TIMIT_VD_MAX_SAMPLE_CNT    150000
#define TIMIT_TE_MAX_SEQUENCE_CNT    2000
#define TIMIT_TE_MAX_SAMPLE_CNT    600000

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



/* Trains a multi layer LSTM followed by Dense layer to recognize
 * speech phonemes from timit dataset.
 *
 * If loadmodel is not NULL, it points to the name of a file of a previously
 * created model, that will be used for further training and inference; 
 * otherwise a new blank model will be created and trained.
 *
 * If storemodel is not NULL, it points to the name of a file, where the 
 * model (further) trained by this function will be stored.
 *
 * layers array contains the size of each layer. One additional output layer
 * is implied.  So for example if layers contains two elements, 128 and 64, 
 * two LSTM layers will be created with size of 128 and 64, followed by 
 * a Dense layer of size 40 (number of phonemes). Note that if a model
 * is loded from a file this parameter is ignored.
 *
 * layers_cnt is the number of entries in layers array.
 *
 * if ctc_mode is not zero, trains model using ctc loss function, and decodes
 * using beam search. Otherwise, trains model using cross entropy loss, and
 * collapses identical labels while decoding.
 *
 * The remaining parameters are passed to model_compile() and model_fit()
 */
int timit_lstm_dense_classification(
                        const char* loadmodel, const char* storemodel,
                        int layers[], int layers_cnt, 
                        int ctc_mode, char* optimizer, int batch_size,
                        float learning_rate, 
                        float weight_decay, 
                        int epochs, char* schedule)
{
    char datetimebuf[20];
    ctc_mode = (ctc_mode) ? 1: 0;
    const char* loss_func = (ctc_mode)?"ctc":"cross-entropy";
    printf("\nTrains a multi layer LSTM followed by Dense layer to predict the\n");
    printf("classes of samples from the TIMIT dataset\n\n");
    printf("Run 'timit -h' to list program options\n\n");
    printf("Training with default parameters may take a few hours\n\n");
    const int Dr = FEAT_CNT;            /* Input vectors dimension          */
    const int D = EXPENDED_FEAT_CNT;    /* Expended input vectors dimension */
    const int N = REDUCED_PHONEME_CNT;  /* Output vector dimension          */
    int L = layers_cnt + 1;             /* Number of layers                 */
    int B = batch_size;                 /* Batch size                       */

    MODEL* m;
    if (loadmodel != NULL) {
        m = load_model(loadmodel);
        if (m == NULL)
            return 0;
        L = m->num_layers;
        if (B >= 0) /* use batch size specified on command line */
            model_set_batch_size(m,B);
        else /* use original model batch size */
            B = m->batch_size;
        layers_cnt = m->num_layers - 1;
        for (int i = 0; i < layers_cnt; i++) {
            LAYER l = m->layer[i];
            switch(l.type) {
                case 'd': layers[i] = l.dense->S; break;
                case 'l': layers[i] = l.lstm->S; break;
            }
        }
    }
    else {
        /* Create Model (to process multiple batches of B samples each) */
        if (B < 0)
            B = -B;
        m = model_create(L,B,D,1,1); /* Notice adding bias to input */
        model_add(m,lstm_create(layers[0],"sigmoid",1),"lstm"); 
        for (int i = 1; i < L - 1; i++)
            model_add(m,lstm_create(layers[i],"sigmoid",1),"lstm");
        model_add(m,dense_create(N,"softmax"),"dense");
        model_compile(m,loss_func,optimizer);
    }

    printf("%d layers (including output layer) ",L);
    for (int i = 0; i < layers_cnt; i++)
        printf("%d,",layers[i]);
    printf("%d.\n",N);
    printf("Input dimension %d. Expended input dimension %d. Batch size %d.\n",
                                                                       Dr,D,B);
    printf("%d epochs, ",epochs);
    if (schedule != NULL)
        printf("learning rate schedule %s \n",schedule);
    else
        printf("learning rate %g, weight decay %g \n",learning_rate,weight_decay);
    printf("Using %s loss function\n\n",loss_func);


    /* Create file name stem for this run */
    char ls[40];    
    for (int i = 0, a = 0, n = layers_cnt; i < n; i++)
        a += snprintf(ls + a,sizeof(ls) - a,"%d%s",layers[i],((i<n-1)?"_":""));
    char fns[100];
    snprintf(fns,sizeof(fns),"e%d-b%d-r%g-w%g-L%s-pid-%d",
             epochs,batch_size,learning_rate,weight_decay,ls,getpid());
             
    typedef float (*ArrMD)[D];
    typedef float (*ArrMN)[N];
    typedef int (*VecM);
    typedef int (*VecS);

    int STr = TIMIT_TR_MAX_SEQUENCE_CNT;
    int MTr = TIMIT_TR_MAX_SAMPLE_CNT;
    ArrMD xTr = allocmem(MTr,D,float);  /* TIMIT training dataset           */
    VecS  sTr = allocmem(STr,1,int);    /* TIMIT training sequence lengths  */
    VecM  yTrc = allocmem(MTr,1,int);   /* Training (true) labels           */
    ArrMN yTrt = allocmem(MTr,N,float); /* Training label one-hot vectors   */
    int SVd = TIMIT_VD_MAX_SEQUENCE_CNT;
    int MVd = TIMIT_VD_MAX_SAMPLE_CNT;
    ArrMD xVd = allocmem(MVd,D,float);  /* TIMIT validation dataset         */
    VecS  sVd = allocmem(SVd,1,int);    /* TIMIT validation sequence lengths*/
    VecM  yVdc = allocmem(MVd,1,int);   /* Validation (true) labels         */
    ArrMN yVdt = allocmem(MVd,N,float); /* Validation label one-hot vectors */
    int STe = TIMIT_TE_MAX_SEQUENCE_CNT;
    int MTe = TIMIT_TE_MAX_SAMPLE_CNT;
    ArrMD xTe = allocmem(MTe,D,float);  /* TIMIT test dataset           */
    VecS  sTe = allocmem(STe,1,int);    /* TIMIT test sequence lengths  */
    VecM  yTec = allocmem(MTe,1,int);   /* Test (true) labels           */
    ArrMN yTet = allocmem(MTe,N,float); /* Test label one-hot vectors   */

    int cnt;

    /* Read data */
    printf("%s Loading data...\n",date_time(datetimebuf));
    cnt = read_feature_files(timit_tr_data_dir,
                             timit_tr_file_list,STr,sTr,MTr,xTr,yTrc);
    if (cnt == 0)
        return 0;
    /* Update sTr, MTr with actual values */
    STr = cnt;
    cnt = 0;
    for (int i = 0; i < STr; i++)
        cnt += sTr[i];
    MTr = cnt;
    /* Each sequence contains multiple phonemes, find how many in total */
    int PTr = count_phoneme(yTrc,MTr);

    cnt = read_feature_files(timit_vd_data_dir,
                             timit_vd_file_list,SVd,sVd,MVd,xVd,yVdc);
    if (cnt == 0)
        return 0;
    /* Update sVd, MVd with actual values */
    SVd = cnt;
    cnt = 0;
    for (int i = 0; i < SVd; i++)
        cnt += sVd[i];
    MVd = cnt;
    /* Each sequence contains multiple phonemes, find how many in total */
    int PVd = count_phoneme(yVdc,MVd);

    cnt = read_feature_files(timit_te_data_dir,
                             timit_te_file_list,STe,sTe,MTe,xTe,yTec);
    if (cnt == 0)
        return 0;
    /* Update sTe, MTe with actual values */
    STe = cnt;
    cnt = 0;
    for (int i = 0; i < STe; i++)
        cnt += sTe[i];
    MTe = cnt;
    /* Each sequence contains multiple phonemes, find how many in total */
    int PTe = count_phoneme(yTec,MTe);

    printf("%d training sequences, %d phonemes, %d samples\n",STr,PTr,MTr);
    printf("%d validation sequences, %d phonemes, %d samples\n\n",SVd,PVd,MVd);
    printf("%d test sequences, %d phonemes, %d samples\n\n",STe,PTe,MTe);
        
    /* Encode yc as one-hot vectors */
    onehot_encode(yTrc,yTrt,MTr,N);
    onehot_encode(yVdc,yVdt,MVd,N);
    onehot_encode(yTec,yTet,MTe,N);

    float losses[epochs];
    float accuracies[epochs];
    float v_losses[epochs];
    float v_accuracies[epochs];

    if (epochs > 0) {
        printf("%s Training...\n",date_time(datetimebuf));
        char kwargs[512];
        snprintf(kwargs,sizeof(kwargs),"schedule=%s verbose=2",schedule);
        model_fit(m,xTr,yTrt,sTr,STr,
                    xVd,yVdt,sVd,SVd,
                    epochs,learning_rate,weight_decay,
                    losses,accuracies,v_losses,v_accuracies,kwargs);
                    
    }

    if (storemodel != NULL) 
        store_model(m,storemodel);

    printf("%s Testing...\n",date_time(datetimebuf));
    int nc = N;     /* Number of classes (inc. slience/blank)       */
    int off;        /* Current sequence offset within test data     */
    float mcnt;     /* Number of matching raw labels                */
    /* variables for similarity calculations */
    int dist1;      /* Raw label sequences edit distance (numerator)*/
    int dist2;      /* Phoneme sequences edit distance              */
    int dist3;      /* Beam search phoneme sequences edit distacne  */
    int len1;       /* Raw label sequences edit length (dnominator) */
    int len2;       /* Phoneme sequences edit length                */
    int len3;       /* Beam search phoneme sequences edit length    */
    int i, j;
    mcnt = 0;
    dist1 = dist2 = dist3 = 0;
    len1 = len2 = len3 = 0;

    int cm[N][N]; /* Confusion matrix */
    memset(cm,0,N * N * sizeof(int));

    int sTeMax = 0;
    for (i = 0; i < STe; i++)
        if (sTeMax < sTe[i])
            sTeMax = sTe[i];
    
    ArrMN yp = allocmem(sTeMax,N,float); /* Predicted probabilities */
    VecS ypc = allocmem(sTeMax,1,int);   /* Predicted labels        */
    VecS ytc = allocmem(sTeMax,1,int);   /* True labels             */
    
    for (int i = 0; i < nc; i++)
        for (int j = 0; j < nc; j++)
            cm[i][j] = 0;
     
    for (i = 0, off = 0; i < STe; off += sTe[i++]) {
        printf("\r%3d sequences out of %d %3d%%",i,STe,off * 100 / MTe);
        fflush(stdout);
        memcpy(ytc,yTec + off,sTe[i] * sizeof(int));
        model_predict(m,xTe + off,yp,sTe[i]);
        onehot_decode(yp,ypc,sTe[i],N); 
        /* Count raw matches */
        for (j = 0; j < sTe[i]; j++)
            if (ytc[j] == ypc[j])
                mcnt++;

        /* Calculate edit distance of raw predictions */
        dist1 += edit_dist(ypc,sTe[i],ytc,sTe[i]);
        len1 += sTe[i];

        /* Convert labels to phonemes: 
         * merge repeated labels, remove silence/blanks 
         */
        int ytc_len = dedup_labels(ytc,sTe[i],SIL);
        int ypc_len = dedup_labels(ypc,sTe[i],SIL);

        /* Calulate edit distance of phonemes */
        dist2 += edit_dist(ypc,ypc_len,ytc,ytc_len);
        len2 += (ytc_len > ypc_len) ? ytc_len : ypc_len;
  
        /* Performs beam search to find the most probable sequences of lables */
        int beamwidth = 3;
        int timesteps = sTe[i];
        int sequences[beamwidth][timesteps + 1];
        float scores[beamwidth];

        beam_search(yp,timesteps,nc,beamwidth,sequences,scores);
        ypc_len = dedup_labels(sequences[0],timesteps,SIL);
        memcpy(ypc,sequences[0],ypc_len * sizeof(int));

        /* Align beam searched phonemes with true phonems;
         * calulate edit distance 
         */
        int rlen = ((ytc_len > ypc_len) ? ytc_len : ypc_len) * 2;
        int ypc2[rlen];
        int ytc2[rlen];
        dist3 += alignseq(ypc,ypc_len,ytc,ytc_len,ypc2,ytc2,rlen,SIL);
        len3 += (ytc_len > ypc_len) ? ytc_len : ypc_len;

        /* Update the confusion matrix */
        for (int i = 0; i < rlen; i++)
            if (ytc2[i] != SIL || ypc2[i] != SIL)
                cm[ytc2[i]][ypc2[i]]++;
            else
                break;
    }
    printf("\r%3d sequences out of %d %3d%%\n",i,STe,off * 100 / MTe);
    printf("%s Testing completed \n",date_time(datetimebuf));
    printf("Accuracy (labels) %5.3f\n",mcnt / MTe);
    printf("Average similarity (label edit distance) %5.3f\n",
                                                1.0 - ((float) dist1) / len1);
    printf("Average similarity (phoneme edit distance) %5.3f\n",
                                                1.0 - ((float) dist2) / len2);
    printf("Average similarity (with beam search) %5.3f\n",
                                                1.0 - ((float) dist3) / len3);
    
    char cmfn[256];
    snprintf(cmfn,sizeof(cmfn),"cm-%s.csv",fns);
    printf("Writing confusion matrix to %s\n",cmfn);
    {
        FILE* fp = fopen(cmfn,"wb");
        for (int j = 0; j < nc; j++)
            fprintf(fp,",%s",reduced_phoneme_names[j]);
        fprintf(fp,"\n");
        for (int i = 0; i < nc; i++) {
            fprintf(fp,"%s",reduced_phoneme_names[i]);
            for (int j = 0; j < nc; j++)
                fprintf(fp,",%d",cm[i][j]);
            fprintf(fp,"\n");
        }
       fclose(fp);   
    }
    
#ifdef HAS_PLOT
    {
        #include "../plot/plot.h"
        char* title = "Confusion Matrix";
        plot_cm(cm,nc,reduced_phoneme_names,
               epochs,losses,accuracies,v_losses,v_accuracies,title,"circles");
    }
#else
    (void) accuracies;
    (void) losses;
    (void) v_accuracies;
    (void) v_losses;
    printf("\n");
#endif
    freemem(yp);
    freemem(ypc);
    freemem(ytc);
    freemem(xTr);
    freemem(sTr);
    freemem(yTrc);
    freemem(yTrt);
    freemem(xTe);
    freemem(sTe);
    freemem(yTec);
    freemem(yTet);
    model_free(m);
    return 1;
}

int main(int argc, char** argv)
{
    const char* usage = 
        "Usage: timit [-h] [-e <epochs>]                                \n"
        "             [-r <learning rate>] [-w weight decay]            \n"
        "             [-b <batch size>] [-L 's1 s2 ...']                \n"
        "             [-l <model file>] [-s <model file>]               \n"
        "             [ -ctc | -cross-entropy ]                         \n"
        "                                                               \n"
        " -L: LSTM layer specification. One additional output layer     \n"
        "     is implied. So for example -L '126 64' specifies two      \n"
        "     LSTM layers followed by a Dense output layer.             \n"
        "     Notice that the specification is quoted.                  \n"
        " -l: Load model from file and continue to train for number     \n"
        "     of epochs specified by -e; ignore -L option.              \n"
        " -s: Store model in file at the end of training                \n"
        "\n";

    int epochs = 21, bsize = -128; /* Nagative value indicates default value */
    float lr = 0.001, wd = 0.01;
    char *sch = "12:0.001:0.01,6:0.0001:0.01,3:0.00001:0";
    char *loadfile = NULL, *storefile = NULL;
    #define maxlyrcnt 5
    int lyrcnt = 3;
    int layers[maxlyrcnt+1] = {128,128,128};
    int ctc_mode = 1;
    int opt;
    while ((opt = getopt(argc, argv, "e:r:w:b:l:s:L:c:h")) != -1) {
        switch (opt) {
            case 'h': printf(usage); exit(0);
            case 'e': epochs = atoi(optarg); break;
            case 'l': loadfile = optarg; break;
            case 's': storefile = optarg; break;
            case 'b': bsize = atoi(optarg); break;
            case 'r': lr = atof(optarg); break;
            case 'w': wd = atof(optarg); break;
            case 'L': 
                lyrcnt = sscanf(optarg,"%d %d %d %d %d %d",&layers[0],
                      &layers[1],&layers[2],&layers[3],&layers[4],&layers[4]);
                if (lyrcnt > maxlyrcnt) {
                    fprintf(stderr,"timit: too many LSTM layers\n");
                    printf(usage);
                    exit(-1);
                }
            break;
            case 'c':
                if (strcmp(optarg,"tc") == 0)
                    ctc_mode = 1;
                else
                if (strcmp(optarg,"ross-entropy") == 0)
                    ctc_mode = 0;
                else {
                    fprintf(stderr,"%s: invalid option -- 'c%s'\n",argv[0],optarg);
                    printf(usage);
                    exit(-1);
                }
            break;
            case '?':
            default: 
                printf(usage);
                exit(-1);
        }
    }
    int ok;
    init_lrng(42);
    ok = timit_lstm_dense_classification(loadfile,storefile,layers,lyrcnt,
                                      ctc_mode,"adamw",bsize,lr,wd,epochs,sch);
    return (ok) ? 0 : -1;
}
