/* Copyright (c) 2023-2024 Gilad Odinak */

/* This program implements vanilla word2vec based on mikolov's paper.
 * The input is a file containing a list of files. Each of these files
 * contains some text.
 */
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <getopt.h>
#include <omp.h>
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

const char* usage =
"Usage: word2vec [options]\n"
"Options:\n"
"  -h                Show this help message\n"
"  -b <batch_size>   Set batch size (default 16)\n"
"  -c <context_size> Set context size (must be even, default 8)\n"
"  -d <embedding_dim>Set embedding dimension (default 100)\n"
"  -e <num_epochs>   Set number of epochs (default 5)\n"
"  -i <train_file>   Set training files list path\n"
"  -o <output_file>  Set output embedding file path\n"
"  -r <learning_rate>Set starting learning rate (default 0.05)\n"
"  --rd=<f>          Learning rate decay (default 0.8)\n"
"  --vocab-size=<n>  Limit vocabulary size\n"
"  --vocab-coverage=<f> Limit vocabulary coverage (default 0.95)\n"
"  --print-vocab     Print vocabulary and exit\n"
;

/* Creates contexts for a text's words. one context per word.
 *
 * sw  - array of text word indices, in the order of the text
 * swc - number of entries in sw
 * cxt - array that receives the contexts (output); it has swc rows
 * cs  - context size; number of word indices in a context, columns in cxt.
 *       Must be an even number.
 *
 * Each context is stored in a row of cxt array. Assumes cs is an even number.
 * The context consist of the indices of cs/2 words preceding the context's
 * target word, followed by the indices of cs/2 words that come after it,
 * in order,
 */
void text2cxt(int* sw, int swc, fArr2D cxt_, int cs)
{
    typedef float (*ArrCS)[cs];
    ArrCS cxt = (ArrCS) cxt_;

    fltclr(cxt,swc * cs); /* Set all elements to the pad (index 0) value */

    int m = cs / 2;
    for (int i = 0; i < swc; i++) {
        for (int j = m, k = i + 1; j < cs && k < swc; j++, k++)
            cxt[i][j] = sw[k];
        for (int j = m - 1, k = i - 1; j >= 0 && k >= 0; j--, k--)
            cxt[i][j] = sw[k];
    }
}

/* Swaps row i with row j of array a[M][N]
 */
static inline void swap_rows(fArr2D a_, int M, int N, int i, int j)
{
    typedef float (*ArrMN)[N];
    ArrMN a = (ArrMN) a_;
    (void) M;
    float t;
    for (int k = 0; k < N; k++) {
        t = a[i][k];
        a[i][k] = a[j][k];
        a[j][k] = t;
    }
}

/* Shuffles the rows of arrays a[M][N], and l[M][1]
 */
static void shuffle_samples(fArr2D a, fArr2D l, int M, int N)
{
    for (int i = M - 1; i > 0; i--) {
        int j = (int) urand(0.0,1.0 + i);
        swap_rows(a,M,N,i,j);
        swap_rows(l,M,1,i,j);
    }
}

static inline float clip_grad(float g)
{
    const float gmax = 5;
    const float gmin = 1e-6;
    float m = fabsf(g);
    if (m > gmax)
        g = (g > 0) ? gmax : -gmax;
    else
    if (m < gmin)
        g = (g > 0) ? gmin : -gmin;
    return g;
}

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

/* Creates and initializes output weight matrix for negative sampling */
fArr2D create_output_weights(int vocab_size, int embedding_dim)
{
    typedef float (*ArrDE)[embedding_dim];
    ArrDE Wo = allocmem(vocab_size,embedding_dim,float);
    float scale = sqrt(2.0 / (vocab_size + embedding_dim));
    for (int i = 0; i < vocab_size ; i++)
        for (int j = 0; j < embedding_dim; j++)
            Wo[i][j] = nrand(0.0,scale);
    return Wo;
}

/* Applies the sigmoid activation function to a single value.
 */
static inline float sigmoid1(float x)
{
    if (x > 100) return 1;
    if (x < -100) return 0;
    return 1.0 / (1.0 + expf(-x));
}

float negative_sampling_loss(
    fArr2D h_,          /* [B][E] context embeddings */
    fArr2D labels,      /* [B][1] target word indices */
    fArr2D gEmb_,       /* [B][E] gradients w.r.t. input embeddings */
    fArr2D Wo_,         /* [D][E] output weights */
    fArr2D gWo_,        /* [D][E] gradients w.r.t. output weights */
    int B,              /* batch size */
    int E,              /* embedding dimension */
    int D,              /* vocab size */
    int* dist_table,    /* unigram table for negative sampling */
    int dist_table_size,
    int nsamples        /* number of negative samples */
)
{
    typedef float (*ArrBE)[E];
    typedef float (*ArrDE)[E];

    ArrBE h = (ArrBE) h_;
    ArrBE gEmb = (ArrBE) gEmb_;
    ArrDE Wo = (ArrDE) Wo_;
    ArrDE gWo = (ArrDE) gWo_;

    float loss = 0;

    // Clear gradients for embeddings and output weights
    fltclr(gEmb, B * E);
    fltclr(gWo, D * E);

    for (int i = 0; i < B; i++) {
        int target = (int) ((float *)labels)[i];
        float dot, p, grad;

        // Positive sample dot product
        dot = 0;
        for (int j = 0; j < E; j++)
            dot += Wo[target][j] * h[i][j];

        p = sigmoid1(dot);
        loss -= logf(p + 1e-8);
        grad = p - 1.0;

        // Gradient updates for positive sample
        for (int j = 0; j < E; j++) {
            gWo[target][j] += grad * h[i][j];
            gEmb[i][j]   += grad * Wo[target][j];
        }

        // Negative samples
        for (int k = 0; k < nsamples;) {
            int inx = (int)urand(0,dist_table_size);
            int neg = dist_table[inx];
            if (neg == target)
                continue;

            // Negative sample dot product
            dot = 0;
            for (int j = 0; j < E; j++)
                dot += Wo[neg][j] * h[i][j];

            p = sigmoid1(-dot);
            loss -= logf(p + 1e-8);
            grad = 1.0 - p;

            // Gradient updates for negative samples
            for (int j = 0; j < E; j++) {
                gWo[neg][j] += grad * h[i][j];
                gEmb[i][j] += grad * Wo[neg][j];
            }
            k++;
        }
    }
    return loss;
}

/* Compare two word frequency values - used with qsort to order
 * words based on their frequency in descending order,
 * that is, most frequent word first.
 */
int qsort_compare_word_freq(const void *a, const void *b)
{   /* WRDFRQ declared in newsfile.h */
    if (((WRDFRQ *)b)->cnt > ((WRDFRQ *)a)->cnt) return 1;
    if (((WRDFRQ *)b)->cnt < ((WRDFRQ *)a)->cnt) return -1;
    return 0;
}

/* Stores cosine similarity of a word to a reference word - used with qsort */
typedef struct wrdsim_s {
    int wrdinx;
    float cossim;
} WRDSIM;

/* Compare two consine similarity values - used with qsort to order
 * words that are similar to a reference word in descending order,
 * that is, most similar (higest cosine similarity value) first.
 */
int qsort_compare_similarity(const void *a, const void *b)
{
    if (((WRDSIM *)b)->cossim > ((WRDSIM *)a)->cossim) return 1;
    if (((WRDSIM *)b)->cossim < ((WRDSIM *)a)->cossim) return -1;
    return 0;
}

void shuffle_list(char** list, int cnt)
{
    if (cnt <= 1) return;
    for (int i = cnt - 1; i > 0; i--) {
        int j = (int) urand(0,i + 1);
        char* temp = list[i];
        list[i] = list[j];
        list[j] = temp;
    }
}

int main(int argc, char** argv)
{
    char* data_dir = "data/news/data"; /* Input */
    char* tr_file = "data/news/tr_files.lst"; /* Input */
    char* stopwords_file = "data/news/stopwords.txt"; /* Input */
    char* embedding_file = "word2vec.test.model"; /* Output */
    float vocab_coverage = 0.95; /* 95% */
    int vocab_size = 0; /* Default: size derived from vocab_coverage */
    int embedding_dim = 100;
    int batch_size = 16;
    int cxt_size = 8;
    int num_epochs = 5;
    float learning_rate = 0.05;
    float learning_rate_decay = 0.8;
    int print_vocab = 0;
    int max_vocab = 3000000;   /* Set to 3 x expected number of unique words */
    int hash_mem = 10000000;   /* hashmap will increase this value as needed */
    int max_file_words = 1000000; /* Maximum number of words per file        */

    int opt;
    while ((opt = getopt(argc,argv,"b:c:d:e:i:o:r:-:h")) != -1) {
        switch (opt) {
            case 'h': printf(usage); exit(0);
            case 'b': batch_size = atoi(optarg); break;
            case 'c': cxt_size = atoi(optarg); break;
            case 'd': embedding_dim = atoi(optarg); break;
            case 'e': num_epochs = atoi(optarg); break;
            case 'i': tr_file = optarg; break;
            case 'o': embedding_file = optarg; break;
            case 'r': learning_rate = atof(optarg); break;
            case '-':
                if (strncmp(optarg,"rd=",3) == 0)
                    learning_rate_decay = atof(optarg+3);
                else
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
                printf(usage);
                exit(-1);
        }
    }

    if (cxt_size % 2) {
        fprintf(stderr,"word2vec: context size must be an even number. got -c %d\n",cxt_size);
        exit(-1);
    }

    printf("\n");
    printf("Trains an embedding layer to create word embeddings using\n");
    printf("Continuous Bag of Words (CBOW) method\n");
    printf("context size = %d, embedding dim = %d, batch size = %d\n"
           "%d epochs, learning_rate = %g learning_rate_decay = %g\n",
           cxt_size,embedding_dim,batch_size,
           num_epochs,learning_rate,learning_rate_decay);
    fflush(stdout);

    int tot_file_cnt = 0; /* Total number of files    */
    int tot_word_cnt = 0; /* Total number of words    */
    int stop_cnt = 0;     /* Number of stop words (including pad) */

    /* Stop words index */
    HASHMAP* stop_hmap = hashmap_create(max_vocab,hash_mem);
    hashmap_str2inx(stop_hmap,"",1); /* Reserve first entry (inx 0) for pad */
    stop_cnt++; /* Count pad as a stop word */

    printf("Loading stop words\n");
    stop_cnt += process_news_file(stopwords_file,NULL,
                                  stop_hmap,1,max_vocab,NULL,NULL,0);

    printf("Creating vocabulary from dataset\n");
    fflush(stdout);
    HASHMAP* hmap = hashmap_create(max_vocab,hash_mem);
    hashmap_str2inx(hmap,"",1);
    tot_word_cnt++;
    WRDFRQ* word_freq = allocmem(max_vocab,1,WRDFRQ);

    int num_files = 0;
    char** file_list = read_news_file_list(tr_file,data_dir,&num_files);
    if (file_list == NULL || num_files == 0) {
        fprintf(stderr,"Failed to read data files list from '%s'\n",tr_file);
        return -1;
    }
    for (int i = 0; i < num_files; i++) {
        tot_file_cnt++;
        tot_word_cnt += process_news_file(file_list[i],data_dir,
                                          hmap,1,max_vocab,word_freq,NULL,0);
    }

    printf("Dataset: %d files, %d words, ",tot_file_cnt,tot_word_cnt);
    printf("%d unique words, %d stop words\n",hmap->map_used,stop_cnt - 1);
    printf("%d bytes of word storage memory used\n",hmap->mem_used);
    fflush(stdout);

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
    printf("The vocabulary covers %d (%2.0f%%) of dataset words\n",
                                       word_cnt,100 * vocab_coverage);

    /* Create new vocabulary index of only the retained words,
     * in order of their frequency in the dataset.
     */
    printf("Creating vocabulary of %d words\n",vocab_size);
    fflush(stdout);
    HASHMAP* hmap2 = hashmap_create(vocab_size * 3,hmap->mem_used);
    hashmap_str2inx(hmap2,"",1); /* Reserve first entry (index 0) for pad */

    /* Add words up to vocabulary size (pad already included) */
    for (int i = 0, j = 0; i < vocab_size - 1 && j < vocab_size - 1; i++) {
        const char* wrd = hashmap_inx2str(hmap,word_freq[i].inx);
        if (strlen(wrd) == 0)
            continue;
        int inx = hashmap_str2inx(hmap2,wrd,1);
        word_freq[j++].inx = inx;
    }
    hashmap_free(hmap);
    hmap = hmap2;
    hmap2 = NULL;

    printf("Calculate word frequencies\n");
    for (int i = 0; i < vocab_size; i++) {
        word_freq[i].frq = ((float) word_freq[i].cnt) / word_cnt;
    }
    
    printf("Creating vocabulary distribution table\n");
    fflush(stdout);
    float dist_compress = 0.75;
    int dist_scale = 10;
    int dist_table_size = 0;
    /* Calculate requried size for all tokens (excluding pad) */
    for (int i = 1; i < vocab_size; i++) {
        int freq = word_freq[i].cnt;
        dist_table_size += (int)(pow(freq,dist_compress)/dist_scale) + 1;
    }
    int* dist_table = allocmem(dist_table_size,1,int);
    printf("Distribution table size %d\n",dist_table_size);

    /* Populate table, note pad (i == 0) is ecluded */
    for (int i = 1, j = 0; i < vocab_size; i++) {
        int inx = word_freq[i].inx;
        int freq = word_freq[i].cnt;
        int rpt = (int)(pow(freq,dist_compress)/dist_scale) + 1;
        for (int k = 0; j < dist_table_size && k < rpt; k++)
            dist_table[j++] = inx;
    }
    
    if (print_vocab) {
        printf("ord   index count word\n");
        for (int i = 0; i < vocab_size; i++) {
            const char* wrdstr = hashmap_inx2str(hmap,word_freq[i].inx);
            printf("%5d %5d %5d %-16s\n",i,word_freq[i].inx,word_freq[i].cnt,wrdstr);
        }
        printf("ord   index word\n");
        for (int i = 0; i < dist_table_size; i++) {
            const char* wrdstr = hashmap_inx2str(hmap,dist_table[i]);
            printf("%5d %5d %-16s\n",i,dist_table[i],wrdstr);
        }
        return 0;
    }

    printf("\n");
    float start_time = current_time();
    printf("Creating word embeddings\n");
    fflush(stdout);

    /* Create and initialize layers */
    EMBEDDING* embedding = embedding_create(embedding_dim,cxt_size,0);
    embedding_init(embedding,vocab_size,batch_size);
    /* Negative sampling only uses an output weights matrix */
    fArr2D Wo = create_output_weights(vocab_size,embedding_dim);

    /* Allocate memory for gradients */
    fArr2D dy;  /* Gradients with respect to the input  */
    dy = allocmem(embedding->B,embedding->E,float);
    fArr2D gWx[2]; /* Gradients with respect to the weights */
    gWx[0] = allocmem(embedding->D,embedding->E,float);
    gWx[1] = allocmem(vocab_size,embedding_dim,float);

    int* file_words = allocmem(1,max_file_words,int);
    int file_cnt;

    for (int epoch = 1; epoch <= num_epochs; epoch++ ) {
        word_cnt = 0;
        float loss = 0;
        file_cnt = 0;
        shuffle_list(file_list,num_files);
        for (int i = 0; i < num_files; i++) {
            file_cnt++;
            int fwcnt = process_news_file(file_list[i],data_dir,
                              hmap,0,max_vocab,NULL,file_words,max_file_words);

            /* Sub sample frequent words by removing some */
            int cnt = 0;
            for (int i = 0; i < fwcnt; i++) {
                float r = urand(0,1.0);
                float t = 1e-5;
                float f = word_freq[file_words[i]].frq;
                float p = (sqrt(f / t) + 1.0) * (t / f);
                if (p > 1.0) p = 1.0;
                if (r < p)
                    file_words[cnt++] = file_words[i];
            }

            /* Process each word in current file */
            for (int i = 0; i < cnt; i += batch_size) {

                /* Create word contexts, which consist of cxt_size words that
                 * are adjacent to the current word, exist in the vocabulary,
                 * and are not stop words.
                 */
                float contexts[batch_size][cxt_size];
                float labels[batch_size][1];
                int wcnt = batch_size;
                if (i + wcnt >= cnt)
                    wcnt = cnt - i;
                text2cxt(file_words + i, wcnt, contexts, cxt_size);

                for (int j = 0, k = i; k < i + wcnt; j++, k++) {
                    labels[j][0] = file_words[k];
                    if (labels[j][0] >= vocab_size) {
                        printf("labels[%d] %g file_word[%d]\n",j,labels[j][0],k);
                        exit(1);
                    }
                }

                shuffle_samples(contexts,labels,wcnt,cxt_size);

                if (wcnt < batch_size) {
                    for (int j = wcnt; j < batch_size; j++) {
                        fltclr(contexts[j],cxt_size);
                        labels[j][0] = 0;
                    }
                }

                /* Forward pass */
                fArr2D yp = embedding_forward(embedding,contexts,0);

                loss += negative_sampling_loss(yp,labels,dy,Wo,gWx[1],
                                               batch_size,embedding_dim,vocab_size,
                                               dist_table,dist_table_size,10);

                /* backward pass */
                embedding_backward(embedding,dy,contexts,gWx[0],NULL,0);

                /* Update weights */
                update(embedding->Wx,gWx[0],embedding->D,embedding->E,learning_rate);
                update(Wo,gWx[1],embedding->D,embedding->E,learning_rate);

                word_cnt += wcnt;
                int pct = (tot_file_cnt >= 1) ? (((float)file_cnt / tot_file_cnt) * 100) : 100;
                int seconds = (int) elapsed_time(start_time);
                int sec = seconds % 60;
                int min = (seconds / 60) % 60;
                int hours = seconds / 3600;
                printf("epoch %2d lr %5.3f loss %6.4f %3d%% "
                       "(file %d of %d, %d words) %d:%02d:%02d\r",
                       epoch,learning_rate,loss / word_cnt,pct,
                       file_cnt,tot_file_cnt,word_cnt,hours,min,sec);
                fflush(stdout);
            }
        }
        printf("\n");
        learning_rate = learning_rate_decay * learning_rate;
    }

    printf("\n");
    printf("Saving word embeddings to %s\n",embedding_file);
    FILE* fp = fopen(embedding_file,"wb");
    if (fp != NULL) {
        printf("#,vocab_size,%d,embedding_dim,%d,"
               "learning_rate,%f,learning_rate_decay,%f,epochs,%d\n",
                vocab_size,embedding_dim,
                learning_rate,learning_rate_decay,num_epochs);

        typedef float (*ArrDE)[embedding->E];
        ArrDE Wx = (ArrDE) embedding->Wx;
        for (int wrdinx = 0; wrdinx < vocab_size; wrdinx++) {
            const char* word = hashmap_inx2str(hmap,wrdinx);
            if (strlen(word) == 0)
                word = "<unk>";
            fprintf(fp,"%d,%s",wrdinx,word);
            for (int j = 0; j < embedding_dim; j++)
                fprintf(fp,",%10.8f",Wx[wrdinx][j]);
            fprintf(fp,"\n");
        }
        fclose(fp);
    }
    else
        fprintf(stderr,"Failed to open file '%s' for write\n",embedding_file);
    printf("\n");
    hashmap_free(hmap);
    embedding_free(embedding);
    freemem(Wo);
    freemem(dy);
    freemem(gWx[0]);
    freemem(gWx[1]);
    freemem(dist_table);
    freemem(word_freq);
    freemem(file_words);
    free_news_file_list(file_list,num_files);
    return 0;
}
