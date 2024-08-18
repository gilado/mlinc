/* Copyright (c) 2023-2024 Gilad Odinak */

/* This program implements vanilla word2vec based on mikolov's paper.
 * The input is a file containing a list of files. Each of these files
 * contains some text.
 */
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <getopt.h>
#include "mem.h"
#include "float.h"
#include "etime.h"
#include "random.h"
#include "array.h"
#include "hash.h"
#include "cossim.h"
#include "newsfile.h"
#include "loss.h"
#include "embedding.h"
#include "dense.h"
#include "model.h"

/* Updates layer's weights in a linear way: 
 * weight = weight - learning_rate * weight_gradient
 * Wx:  weight matrix [D][N]
 * gWx: gradient matrix [D][N]
 * lr:  learning_rate
 */
static void update(fArr2D Wx_, fArr2D gWx_, int D, int N, float lr)
{
    typedef float (*ArrDN)[N];
    ArrDN Wx = (ArrDN) Wx_;
    ArrDN gWx = (ArrDN) gWx_;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < N; j++)
            Wx[i][j] -= lr * gWx[i][j];
}

/* Returns the embedding vector of a word
 * embd - embedding layer
 * wrdinx - the index of the word
 */
float* word_embedding(EMBEDDING* embd, int wrdinx)
{
    if (wrdinx < 0 || wrdinx >= embd->D)
        wrdinx = 0;
    typedef float (*ArrDE)[embd->E];
    ArrDE Wx = (ArrDE)  embd->Wx;
    return Wx[wrdinx];
}

/* Compare two word frequency values - used with qsort to order
 * words based on their frequency in descnding order, 
 * that is, most frequent word first.
 */
int qsort_compare_word_freq(const void *a, const void *b) 
{   /* WRDFRQ declared in newsfile.h */
    float diff = ((WRDFRQ *)b)->cnt - ((WRDFRQ *)a)->cnt;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

/* Stores cosine similarity of a word to a reference word - used with qsort */
typedef struct wrdsim_s {
    int wrdinx;
    float cossim;
} WRDSIM; 

/* Compare two consine similarity values - used with qsort to order
 * words that are similar to a reference word in descnding order, 
 * that is, most similar (higest cosine similarity value) first.
 */
int qsort_compare_similarity(const void *a, const void *b) 
{
    float diff = ((WRDSIM *)b)->cossim - ((WRDSIM *)a)->cossim;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

int main(int argc, char** argv)
{
    char* data_dir = "data/news/data"; /* Input */
    char* tr_file = "data/news/tr_files.lst"; /* Input */
    char* stopwords_file = "data/news/stopwords.txt"; /* Input */
    char* embedding_file = "word2vec.test.model"; /* Output */
    float vocab_coverage = 0.95; /* 95% */
    int vocab_size = 0; /* Default: size dervied from vocab_coverage */
    int embedding_dim = 50;
    int batch_size = 100;
    int cxt_size = 4;
    int num_epochs = 1;
    float learning_rate = 0.01;
    int print_vocab = 0;
    int max_vocab = 3000000;   /* Set to 3 x expected number of unique words */
    int hash_mem = 10000000;   /* hashmap will increase this value as needed */
    int max_file_words = 1000000; /* Maximum number of words per file        */

    int opt;
    while ((opt = getopt(argc,argv,"b:c:d:e:i:o:r:-:h")) != -1) {
        switch (opt) {
            case 'h': printf("usage - later\n"); exit(0);
            case 'b': batch_size = atoi(optarg); break;
            case 'c': cxt_size = atoi(optarg); break;
            case 'd': embedding_dim = atoi(optarg); break;
            case 'e': num_epochs = atoi(optarg); break;
            case 'i': tr_file = optarg; break;
            case 'o': embedding_file = optarg; break;
            case 'r': learning_rate = atof(optarg); break;
            case '-':
                if (strncmp(optarg,"vocab-size=",11) == 0)
                    vocab_size = atoi(optarg+11);
                else
                if (strncmp(optarg,"vocab-coverage=",15) == 0)
                    vocab_coverage = atof(optarg+15);
                else
                if (strncmp(optarg,"print-vocab",11) == 0)
                    print_vocab = 1;
                else
                    goto opterr;
            break;
            case '?': 
            default: 
            opterr:
                fprintf(stderr,"word2vec: syntax error"); 
                printf("usage - later\n");
                exit(-1);
        }
    }

    printf("\n");   
    printf("Trains an embedding layer to create word embeddings using\n");
    printf("Continous Bag of Words (CBOW) method\n");
    printf("context size = %d, embedding dim = %d, batch size = %d\n"
           "%d epochs, learning_rate = %g\n\n",
           cxt_size,embedding_dim,batch_size,num_epochs,learning_rate);

    int tot_file_cnt = 0; /* Total number of files    */
    int tot_word_cnt = 0; /* Total number of words    */
    int stop_cnt = 0;     /* Number of stop words (including pad) */

    /* Vocabulary index */
    HASHMAP* hmap = hashmap_create(max_vocab,hash_mem);
    hashmap_str2inx(hmap,"",1); /* Reserve first entry (index 0) for pad */
    stop_cnt++; /* Count pad as a stop word */

    printf("Loading stop words\n");
    FILE* fp = fopen(stopwords_file,"rb");
    if (fp == NULL) {
        fprintf(stderr,"Failed to open file '%s' for read\n",stopwords_file);
        return -1;
    }
    stop_cnt += process_file(fp,hmap,1,max_vocab,NULL,NULL,0);
    fclose(fp);

    printf("Creating remaining vocabulary from dataset\n");
    WRDFRQ* word_freq = allocmem(max_vocab,1,WRDFRQ);
    FILE* lfp = fopen(tr_file,"rb");
    if (lfp == NULL) {
        fprintf(stderr,"Failed to open file '%s' for read\n",tr_file);
        return -1;
    }
    for (;;) {
        int maxpath = 512;
        char filepath[maxpath * 3];
        strcpy(filepath,data_dir);
        int pfxlen = strlen(filepath);
        if (filepath[pfxlen - 1] != '/')
            strcat(filepath,"/");
        char* filename = filepath + strlen(filepath);
        filename = fgets(filename,maxpath,lfp);
        if (filename == NULL || strlen(filename) == 0)
            break;      /* End of file list */
        if (filename[strlen(filename) - 1] == '\n')
            filename[strlen(filename) - 1] = '\0';
        const char* ext = ".txt";
        int off = strlen(filename) - strlen(ext);
        if (off <= 0 || strcasecmp(filename + off, ext) != 0)
            continue; /* Skip files whose name does not end with ext */
        tot_file_cnt++;
        FILE* fp = fopen(filepath, "rb");
        if (fp == NULL) {
            fprintf(stderr,
                    "Failed to open file '%s' (%d) for read - skipping\n",
                                                       filepath,tot_file_cnt);
            continue;
        }
        tot_word_cnt += process_file(fp,hmap,1,max_vocab,word_freq,NULL,0);
        fclose(fp);
    }
    fclose(lfp);

    printf("Dataset: %d files, %d words, ",tot_file_cnt,tot_word_cnt);
    printf("%d unique words, %d stop words\n",hmap->map_used,stop_cnt - 1);
    printf("%d bytes of word storage memory used\n",hmap->mem_used);

    /* Sort vocabulary words by frequency, descending */
    qsort(word_freq,hmap->map_used,sizeof(WRDFRQ),qsort_compare_word_freq);
    int word_cnt = 0;
    if (vocab_size == 0) {
        /* Calculate how many most frequent vocabulary words are needed
         * to represent vocab_coverage percent of all corpus words
         */
        for (; vocab_size < hmap->map_used; vocab_size++) {
            word_cnt += word_freq[vocab_size].cnt;
            if (word_cnt >= (int) (vocab_coverage * ((float) tot_word_cnt)))
                break;
        }
        vocab_coverage = ((float) word_cnt) / tot_word_cnt;
    }
    else {
        /* Calculate what percentage of all corpus words can be 
         * represented by the vocab_size most frequent vocabulary words
         */
        if (vocab_size > hmap->map_used)
            vocab_size = hmap->map_used;
        for (int vocab_inx = 0; vocab_inx < vocab_size; vocab_inx++)
            word_cnt += word_freq[vocab_inx].cnt;
    }
    vocab_coverage = ((float) word_cnt) / tot_word_cnt;
    printf("Limit vocabulary to %d most frequent words\n",vocab_size);
    printf("The vocabulary covers %2.0f%% of dataset words\n",
                                                100 * vocab_coverage);

    if (print_vocab) {    
        for (int i = 0; i < vocab_size; i++) {
            const char* wrdstr = hashmap_inx2str(hmap,word_freq[i].inx);
            printf("%16s %d\n",wrdstr,word_freq[i].cnt);
        }
        return 0;
    }
    /* Create new vocabulary index of only the retained words,
     * in order of their frequency in the dataset.
     */
    printf("Creating vocabulary of %d words\n",vocab_size);
    HASHMAP* hmap2 = hashmap_create(vocab_size * 3,hmap->mem_used);
    hashmap_str2inx(hmap2,"",1); /* Reserve first entry (index 0) for pad */
    for (int i = 0; i < vocab_size; i++)
        hashmap_str2inx(hmap2,hashmap_inx2str(hmap,word_freq[i].inx),1);
    hashmap_free(hmap);
    hmap = hmap2;
    hmap2 = NULL;
    
    printf("\n");
    float start_time = current_time();
    printf("Creating word embeddings\n");

    /* Create and initialize layers */
    EMBEDDING* embedding = embedding_create(embedding_dim,cxt_size,0);
    embedding_init(embedding,vocab_size,batch_size);
    DENSE* dense = dense_create(vocab_size,"softmax");
    dense_init(dense,embedding_dim,batch_size);

    /* Allocate memory for gradients */
    fArr2D dy[2];  /* Gradients with respect to the inputs  */
    fArr2D gWx[2]; /* Gradients with respect to the weights */
    dy[0] = allocmem(embedding->B,embedding->S,float);
    dy[1] = allocmem(dense->B,dense->S,float);
    gWx[0] = allocmem(embedding->D,embedding->E,float);
    gWx[1] = allocmem(dense->D,dense->S,float);

    int* file_words = allocmem(1,max_file_words,int);

    lfp = fopen(tr_file,"rb");
    if (lfp == NULL) {
        fprintf(stderr,"Failed to open file '%s' for read\n",tr_file);
        return -1;
    }
    word_cnt = 0;
    float loss = 0;
    int file_cnt = 0;
    for (;;) {
        int maxpath = 512;
        char filepath[maxpath * 3];
        strcpy(filepath,data_dir);
        int pfxlen = strlen(filepath);
        if (filepath[pfxlen - 1] != '/')
            strcat(filepath,"/");
        char* filename = filepath + strlen(filepath);
        filename = fgets(filename,maxpath,lfp);
        if (filename == NULL || strlen(filename) == 0)
            break;      /* End of file list */
        if (filename[strlen(filename) - 1] == '\n')
            filename[strlen(filename) - 1] = '\0';
        const char* ext = ".txt";
        int off = strlen(filename) - strlen(ext);
        if (off <= 0 || strcasecmp(filename + off, ext) != 0)
            continue; /* Skip files whose name does not end with ext */
        file_cnt++;
        FILE* fp = fopen(filepath, "rb");
        if (fp == NULL) {
            fprintf(stderr,
                    "Failed to open file '%s' (%d) for read - skipping\n",
                                                            filepath,file_cnt);
            continue;
        }
        int cnt = process_file(fp,hmap,0,max_vocab,
                               NULL,file_words,max_file_words);
        fclose(fp);
        
        /* Process each word in current file */
        for (int i = 0, ii = 0; i < cnt; i += batch_size) {
            /* Create word contexts, which consist of cxt_size words that 
             * are adjacent to the current word, exist in the vocabulary,
             * and are not stop words.
             *
             * i is the index of the start of a batch in the file's words
             * ii is the index of a word in the batch
             * cnt is the number of words in the file
             */
            float contexts[batch_size][cxt_size];
            float labels[batch_size][1];
            for (ii = 0; ii < batch_size && i + ii < cnt; ii++) {
                int m = cxt_size / 2;
                int j; /* index of a word in file's words */
                int k; /* index of a word in a context    */
                for (k = m, j = i + ii + 1; k < cxt_size && j < cnt; j++)
                    if (file_words[j] < stop_cnt)
                        contexts[ii][k++] = file_words[j];
                while (k < cxt_size) /* pad as needed */
                    contexts[ii][k++] = 0;
                for (k = m - 1, j = i + ii - 1; k >= 0 && j >= 0; j--)
                    if (file_words[j] < stop_cnt)
                        contexts[ii][k--] = file_words[j];
                while (k >= 0) /* pad as needed */
                    contexts[ii][k--] = 0;

                labels[ii][0] = file_words[i + ii];
            }
            if (ii < batch_size) {
                fltclr(contexts[ii],(batch_size - ii) * cxt_size);
                fltclr(labels[ii],(batch_size - ii) * 1);
            }
            
            /* Forward pass */
            fArr2D yp[2];
            yp[0] = embedding_forward(embedding,contexts,0);
            yp[1] = dense_forward(dense,yp[0],1);
            loss += sparse_cross_entropy_loss(
                                     yp[1],labels,batch_size,vocab_size);

            /* Backward pass */
            dLdy_sparse_cross_entropy_loss(
                                     yp[1],labels,dy[1],batch_size,vocab_size);
            dense_backward(dense,dy[1],yp[0],gWx[1],dy[0],1);
            embedding_backward(embedding,dy[0],contexts,gWx[0],NULL,0);

            /* Update weights */
            update(embedding->Wx,gWx[0],embedding->D,embedding->E,learning_rate);
            update(dense->Wx,gWx[1],dense->D,dense->S,learning_rate);
            word_cnt += ii;
            int pct = file_cnt / (tot_file_cnt / 100);
            int seconds = (int) elapsed_time(start_time);
            int sec = seconds % 60;
            int min = (seconds / 60) % 60;
            int hours = seconds / 3600;
            printf("loss %5.2f %3d%% "
                   "(file %d of %d, %d words) %d:%02d:%02d\r",
                   loss / word_cnt,pct,
                   file_cnt,tot_file_cnt,word_cnt,hours,min,sec);
            fflush(stdout);
        }                                                    
    }
    fclose(lfp);

    printf("\n\n");
    printf("Saving word embeddings to %s\n",embedding_file);
    fp = fopen(embedding_file,"wb");
    if (fp != NULL) {
        typedef float (*ArrDE)[embedding->E];
        ArrDE Wx = (ArrDE) embedding->Wx;
        for (int i = 0; i < vocab_size; i++) {
            fprintf(fp,"%d,%s",i,hashmap_inx2str(hmap,i));
            for (int j = 0; j < embedding_dim; j++)
                fprintf(fp,",%10.8f",Wx[i][j]);
            fprintf(fp,"\n");
        }
        fclose(fp);
    }
    else
        fprintf(stderr,"Failed to open file '%s' for write\n",embedding_file);
    printf("\n");
    hashmap_free(hmap);
    embedding_free(embedding);
    dense_free(dense);
    freemem(dy[0]);
    freemem(dy[1]);
    freemem(gWx[0]);
    freemem(gWx[1]);
    freemem(word_freq);
    freemem(file_words);
    return 0;
}
