/* Copyright (c) 2023-2024 Gilad Odinak */

/* This program implements vanilla word2vec based on mikolov's paper.
 * The input is a file containing a list of files. Each of these files
 * contains some text. The words in each file are related; the sentences
 * in differenct files are independent.
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#define WDIM 63 // Number of dimensions of word vectors
#define NCXT  5 // Number of words in context (including target word)

//static_assert(NCXT % 2 == 1); // NCXT must be an odd number

#define LEARNING_RATE 0.01
#define NUM_EPOCHS 30

#define MAX_WORDS      15000 // unique words
#define MAX_CONTEXTS  200000 // total number of words
#define MAX_WORD_LEN      31 // maximum nuber of characters in a word

#define HASHTAB_SIZE (MAX_WORDS * 4)

// Stores all words
static char word_index[MAX_WORDS][MAX_WORD_LEN+1];
static int wordcnt = 0;

// Holds indcies of words in word_index table.
static int word_hash_map[HASHTAB_SIZE] = {[0 ... HASHTAB_SIZE-1] = -1};

typedef struct context_S {
    int tgtinx;             // Target word index
    int ctxinx[NCXT - 1];   // Context word indices (-1 if missing)
} CONTEXT;

static CONTEXT contexts[MAX_CONTEXTS];
static int ctxcnt = 0;

// word_embeddings[i] corresponds to word_index[i]
static float word_embeddings[MAX_WORDS][WDIM];

// Adapted from djb2_hash. 
// https://theartincode.stanis.me/008-djb2/
static inline int djb2_hash(const char* word)
{
    unsigned int hash = 5381;
    for (int i = 0; word[i] != '\0'; i++)
        hash = ((hash << 5) + hash) + word[i]; /* hash * 33 + c */
    return (int) (hash % HASHTAB_SIZE); // Index into word_hash_map
}

/* Returns the index of the passed in word, -1 on error.
 */
static int word2index(const char* word)
{
    if (strlen(word) > MAX_WORD_LEN)
        return -1;
    int hinx = djb2_hash(word); // Hash of word
    int first = hinx; 
    for (;;) {
        int winx = word_hash_map[hinx]; // Index into word_index
        if (winx == -1) { // No entry / end of search for this hash
            // Therefore no entry for this word - add it
            if (wordcnt >= MAX_WORDS) // Table full 
                return -1;
            int winx = wordcnt++;
            strcpy(word_index[winx],word);
            word_hash_map[hinx] = winx;
            return winx;
        }
        if (strcmp(word_index[winx],word) == 0) // Already there
            return winx;
        // Linear search
        if (++hinx >= HASHTAB_SIZE) // Wrap around
            hinx = 0;
        if (hinx == first)  // Entire table searched
            return -1;
    }
}
    
/* Returns a pointer to the word indexed by the passed in value,
 * or blank string if the passed in index is out of bounds.
 */
const char* index2word(int inx)
{
    return (inx >= 0 && inx < wordcnt) ? word_index[inx] : "";
}

/* Reads all files listed in the passed in file. 
 * Tokenize the text, stores all unique words in word_index and store the 
 * contexts of all words in these files in contexts.
 *
 * Returns 0 on completions; otherwise returns -1 if the passed in file
 * cannot be read; files listed in the passed in file that cannot be read
 * are ignored (and a message is printed for each such file).
 */
int load_word_data(const char* listfile)
{
    printf("loading corpus - reading file list from %s\n",listfile);
    fflush(stdout);
    FILE* flp = fopen(listfile,"rb");
    if (flp == NULL) {
        fprintf(stderr,"%s: failed to open for read\n",listfile);
        return -1;
    }
    int filecnt = 0;
    for (filecnt = 0; ctxcnt < MAX_CONTEXTS; filecnt++) {
        char buffer[500];
        char *fn = fgets(buffer,sizeof(buffer),flp);
        if (fn == NULL) // End of file list
            break;
        buffer[sizeof(buffer) - 1] = '\0';
        char *eol = index(fn,'\n');
        if (eol != NULL) *eol = '\0';
        FILE* fp = fopen(fn,"rb");
        if (fp == NULL) {
            fprintf(stderr,"%s: failed to open for read - ignoring\n",fn);
            continue;
        }
        printf("\r                                    "
               "                                    \r");
        fflush(stdout);
        printf("\rfile %5d: %s\r",filecnt+1,fn);
        fflush(stdout);
        // Expect one text line per file
        char buffer2[2000];
        char *txt = fgets(buffer2,sizeof(buffer2),fp);
        fclose(fp);
        if (txt == NULL) {
            fprintf(stderr,"%s: file is empty - ignoring\n",fn);
            continue;
        }
        buffer[sizeof(buffer) - 1] = '\0';
        eol = index(txt,'\n');
        if (eol != NULL) *eol = '\0';
        int len = strlen(txt);
        // Remove all non alphanumeric characters, lower case remaining,
        // tokenize into words separated by '\0', trim space.
        int prvspc = 0;
        int i, j;
        for (i = j = 0; txt[i] != '\0'; i++) {
            if (isalnum(txt[i])) {
                txt[j++] = tolower(txt[i]);
                prvspc = 0;
            }
            else
            if (isspace(txt[i])) {
                if (!prvspc && j > 0) {
                    txt[j++] = '\0';
                    prvspc = 1;
                }
            }
            else
            if (txt[i] == '\'') { // Preserves apostroph (REVIEW)
                if (i > 0 && isalpha(txt[i - 1]))
                    txt[j++] = '\'';
            }
        }
        txt[j] = '\0';
        len = j; // Length of tokenized text  

        int ctxwin[NCXT]; // Context window (target word in the middle)
        for (i = 0; i < NCXT; i++) 
            ctxwin[i] = -1;
        int ctxinx = 0;

        // Skip the first two tokens, which are sentence start and end times
        for (i = j = 0; i < len && j < 2; j++)
            i += strlen(txt + i) + 1;
            
        // Process remaining tokens
        while (i < len) {
            ctxwin[ctxinx++] = word2index(txt + i);
            if (ctxinx >= NCXT)
                ctxinx = 0;
            i += strlen(txt + i) + 1; // next token
            if (ctxcnt >= MAX_CONTEXTS) {
                fprintf(stderr,"at file %d (%s): maximum number of contexts reached - ignoring rest of data\n",filecnt + 1,fn);
                break;
            }
            // Find the middle of data in context window
            for (j = NCXT - 1; j >= 0; j--)
                if (ctxwin[j] != -1)
                    break;
            if (j < 0) // context window empty - move on
                continue;
            j /= 2; // Middle
            contexts[ctxcnt].tgtinx = ctxwin[j];
            int jl = j - 1;
            int ju = j + 1;
            int k;
            // Store available context word indcies
            for (k = 0; k < NCXT - 1; k++) {
                if (jl >= 0) {
                    if (ctxwin[jl] != -1)
                        contexts[ctxcnt].ctxinx[k++] = ctxwin[jl];
                    jl--;
                }
                if (ju < NCXT - 1) {
                    if (ctxwin[ju] != -1)
                        contexts[ctxcnt].ctxinx[k++] = ctxwin[ju];
                    ju++;
                }
                if (jl < 0 && ju >= NCXT - 1)
                    break;
             }
             while (k < NCXT - 1)
                 contexts[ctxcnt].ctxinx[k++] = -1;
             ctxcnt++;
        }
    }
    printf("\n");
    fflush(stdout);
    return 0;
}

/* lrng returns a pseudo-random real number uniformly distributed 
 * between 0.0 and 1.0. 
 * Lehmer random number generator - Steve Park & Dave Geyer
 */
static int32_t lrng_seed = 123456789;
static void init_lrng(int seed)
{
    seed = seed %200000000;
    printf("init_lrng seed %d -> %d\n",lrng_seed,seed);
    lrng_seed = seed;
}
static float lrng(void)
{
  const int32_t modulus = 2147483647;
  const int32_t multiplier = 48271;
  const int32_t q = modulus / multiplier;
  const int32_t r = modulus % multiplier;
  int32_t t = multiplier * (lrng_seed % q) - r * (lrng_seed / q);
  lrng_seed = (t > 0) ? t : t + modulus;
  return ((float) lrng_seed / modulus);
}

/* Returns a random number following a uniform distribution
 */
static float urand(float min, float max) 
{
    return lrng() * (max - min) + min;
}

static void train()
{
    float msize = (4.0 * wordcnt + 2.0 * WDIM + 4.0 * wordcnt * WDIM) * sizeof(float);
    printf("Training - required additional memory: %8.0lfKB\n",msize / 1e3);
    float* x = malloc(wordcnt * sizeof(float));
    float* e = malloc(wordcnt * sizeof(float));
    float* u = malloc(wordcnt * sizeof(float));
    float* h = malloc(WDIM * sizeof(float));
    float* t = malloc(WDIM * sizeof(float));
    float (*w1)[WDIM] = malloc(wordcnt * WDIM * sizeof(float));
    float (*dw1)[WDIM] = malloc(wordcnt * WDIM * sizeof(float));
    float (*w2)[wordcnt] = malloc(WDIM * wordcnt * sizeof(float));
    float (*dw2)[wordcnt] = malloc(WDIM * wordcnt * sizeof(float));
    float* yp = malloc(wordcnt * sizeof(float));
    if (x == NULL || e == NULL || u == NULL || h == NULL || t == NULL || 
        w1 == NULL || dw1 == NULL || w2 == NULL || dw2 == NULL || yp == NULL) {
        fprintf(stderr,"Not enough memory - exiting\n");
        exit(1);
    }
    
    // Initialize weight matrices
    init_lrng(time(NULL));
    for (int i = 0; i < wordcnt; i++)
        for (int j = 0; j < WDIM; j++)
            w1[i][j] = urand(-0.5,0.5);
    for (int i = 0; i < WDIM; i++)
        for (int j = 0; j < wordcnt; j++)
            w2[i][j] = urand(-0.5,0.5);
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float loss = 0.0;
        for (int i = 0; i < ctxcnt; i++) {
            printf("epoch %3d, progress %5.1f%%\r",epoch + 1,100.0 * i / ctxcnt);
            fflush(stdout);    

            // Create onehot vector for target word
            for (int j = 0; j < wordcnt; j++)
                x[j] = 0.0;
            x[contexts[i].tgtinx] = 1.0;

            for (int j = 0; j < wordcnt; j++)
                u[j] = 0.0;

            // Forward
            // h = x @ w1
            for (int j = 0; j < WDIM; j++) {
                h[j] = 0.0;
                for (int k = 0; k < wordcnt; k++)
                    h[j] += x[k] * w1[k][j];
            }
            // u = h @ w2
            for (int j = 0; j < wordcnt; j++) {
                u[j] = 0.0;
                for (int k = 0; k < WDIM; k++)
                    u[j] += h[k] * w2[k][j];
            }

            // Softmax
            float m = 0.0; // max(u[])
            for (int j = 0; j < wordcnt; j++)
                if (m < u[j])
                    m = u[j];
            float s = 0.0; // sum(exp(u[] - m))
            for (int j = 0; j < wordcnt; j++) {
                yp[j] = exp(u[j] - m);
                s += yp[j];
            }
            for (int j = 0; j < wordcnt; j++)
                yp[j] /= s;

            // Calculate error
            for (int j = 0; j < wordcnt; j++)
                e[j] = 0.0;
            for (int j = 0; j < NCXT - 1; j++) { // All context word indcies
                int hot = contexts[i].ctxinx[j];
                if (hot == -1) // Empty slot
                    continue;
                // hot is the 1 element in the onehot context word vector
                // subtracting that vector from yp is equivalent to yp[k] 
                // unchanged for all k except where k == hot yp[k] -= 1.0
                // Addding the result of subtraction to the error
                for (int k = 0; k < wordcnt; k++)
                    e[k] += yp[k];
                e[hot] -= 1.0;
            }

            // Backward
            // dw2 = h ⊚ e (outer multiplication)
            for (int j = 0; j < WDIM; j++) 
                for (int k = 0; k < wordcnt; k++)
                    dw2[j][k] = h[j] * e[k];
            // t = w2 @ e
            // dw1 = x ⊚ t 
            for (int j = 0; j < WDIM; j++)
                for (int k = 0; k < wordcnt; k++)
                    t[j] = w2[j][k] * e[k];
            for (int j = 0; j < wordcnt; j++) 
                for (int k = 0; k < WDIM; k++)
                    dw1[j][k] = x[j] * t[k];

            // Update
            for (int j = 0; j < wordcnt; j++)
                for (int k = 0; k < WDIM; k++)
                    w1[j][k] -= LEARNING_RATE * dw1[j][k];
            for (int j = 0; j < WDIM; j++)
                for (int k = 0; k < wordcnt; k++)
                    w2[j][k] -= LEARNING_RATE * dw2[j][k];

            // Calculate loss
            float cnt = 0;
            for (int j = 0; j < NCXT - 1; j++) {
                int hot = contexts[i].ctxinx[j];
                if (hot != -1) {
                    cnt++;
                    loss -= u[hot];
                }
            }
            float sum = 0.0;
            for (int k = 0; k < wordcnt; k++)
                sum += exp(u[k]);
            loss += cnt * log(sum);
        }
        printf("epoch %3d, loss %10.1f\n",epoch + 1,loss);
        fflush(stdout);    
    }
    // Store computed word embeddings
    for (int i = 0; i < wordcnt; i++)
        for (int j = 0; j < WDIM; j++)
             word_embeddings[i][j] = w1[i][j];
    printf("\n");
    fflush(stdout);    
    free(x);
    free(e);
    free(u);
    free(h);
    free(w1);
    free(dw1);
    free(w2);
    free(dw2);
    free(yp);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,"Syntax: word2vec <filelist file>  <embeddings file>\n");
        return -1;
    }
    if (strcmp(argv[1],argv[2]) == 0) {
        fprintf(stderr,"model file and filelist file names must be different\n");
        return -1;
    }

    char* listfile = argv[1];
    load_word_data(listfile);
    printf("%d contexts, %d unique words\n",ctxcnt,wordcnt);
    train();
    printf("writing embeddings\n");
    char* embeddingsfile = argv[2];
    FILE *fp = fopen(embeddingsfile,"wb");
    if (fp == NULL) {
        fprintf(stderr,"failed to open %s for write\n",embeddingsfile);
        return -1;
    }
    
    static char commas[WDIM + 1] = { [0 ... WDIM - 2] = ',', '\0' };
    fprintf(fp,"word,index,ndim,%s\n",commas);
    for (int i = 0; i < wordcnt; i++) {
        fprintf(fp,"%s,%d,%d",word_index[i],i,WDIM);
        for (int j = 0; j < WDIM;j++)
            fprintf(fp,",%11.8f",word_embeddings[i][j]);
        fprintf(fp,"\n");
    }
    fclose(fp);
    printf("done\n");
}
