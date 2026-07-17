/* Copyright (c) 2026 */
/* Character-level language model on the TinyShakespeare corpus.
 *
 * A single program that trains and samples from either a decoder-only
 * Transformer or a stacked LSTM, selectable at run time with --model.
 *
 * Pipeline (shared head, only the middle stack changes):
 *
 *   one-hot[K] --> dense(D,"none")            (learned char embedding)
 *              --> N x transformer(H,T,D,Dff) (causal; RoPE inside MHA)
 *      or      --> N x lstm(D)
 *              --> dense(K,"Softmax")          (vocab logits -> probs)
 *
 * Notes on design (see the library headers):
 *   - The Transformer's MHA applies rotary position embeddings (RoPE)
 *     internally, so no explicit positional features are added to the input;
 *     a plain one-hot per character is enough.
 *   - The MODEL container fixes batch size == sequence length T, so every
 *     training example is exactly T characters and the target is the input
 *     shifted left by one.
 *   - Dropout is 0, so training and inference forward passes are identical.
 *   - Autoregressive sampling uses a fixed-width sliding window of length T:
 *     the whole window is re-run each step and the next character is drawn
 *     from the distribution at the last position. RoPE makes this correct
 *     because positions are relative within the window. The LSTM uses the
 *     same windowed loop (stateful = 0), so a single generate() serves both.
 *
 * Get the corpus (about 1.1 MB) if you don't have it:
 *   https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "mem.h"
#include "random.h"
#include "array.h"
#include "loss.h"
#include "model.h"
#include "dense.h"
#include "transformer.h"
#include "lstm.h"

typedef struct {
    const char* model;   /* "transformer" or "lstm"                    */
    const char* data;    /* path to the corpus                         */
    const char* prompt;  /* seed text for generation                   */
    int   block;         /* sequence length T                          */
    int   dim;           /* model dimension D                          */
    int   ffn;           /* FFN hidden dimension Dff (transformer)     */
    int   heads;         /* attention heads (transformer)              */
    int   layers;        /* number of transformer/lstm layers          */
    int   epochs;        /* training epochs                            */
    float lr;            /* learning rate                              */
    float wd;            /* weight decay                               */
    int   gen;           /* number of characters to generate           */
    float temp;          /* sampling temperature (<=0 => greedy)       */
    int   max_seqs;      /* cap on training+val sequences (0 = all)    */
    float val_frac;      /* fraction of sequences held out             */
    unsigned int seed;   /* RNG seed (0 => time based)                 */
} CONFIG;

static void config_defaults(CONFIG* c)
{
    c->model    = "transformer";
    c->data     = "data/tinyshakespeare/input.txt";
    c->prompt   = "\n";
    c->block    = 64;
    c->dim      = 64;
    c->ffn      = 256;
    c->heads    = 4;
    c->layers   = 2;
    c->epochs   = 20;
    c->lr       = 5e-4f;
    c->wd       = 0.01f;
    c->gen      = 500;
    c->temp     = 0.8f;
    c->max_seqs = 4000;
    c->val_frac = 0.05f;
    c->seed     = 0;
}

static const char* USAGE =
 "Usage: charlm [options]\n"
 "  --model M      transformer | lstm            (default transformer)\n"
 "  --data PATH    corpus file                   (default data/tinyshakespeare/input.txt)\n"
 "  --block T      sequence length               (default 64)\n"
 "  --dim D        model dimension               (default 64)\n"
 "  --ffn F        FFN hidden dim (transformer)  (default 256)\n"
 "  --heads H      attention heads (transformer) (default 4)\n"
 "  --layers N     number of layers              (default 2)\n"
 "  --epochs E     training epochs               (default 20)\n"
 "  --lr R         learning rate                 (default 0.0005)\n"
 "  --wd W         weight decay                  (default 0.01)\n"
 "  --gen G        chars to generate after train (default 500)\n"
 "  --temp X       sampling temperature, <=0 greedy (default 0.8)\n"
 "  --prompt S     seed text for generation      (default newline)\n"
 "  --maxseqs N    cap on #sequences, 0 = all    (default 4000)\n"
 "  --valfrac F    validation fraction           (default 0.05)\n"
 "  --seed N       RNG seed, 0 = time based      (default 0)\n"
 "  -h, --help     print this message\n";

static int parse_args(int argc, char** argv, CONFIG* c)
{
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        #define NEXT() (++i < argc ? argv[i] : (fprintf(stderr,"%s",USAGE),exit(1),""))
        if      (!strcmp(a,"--model"))   c->model    = NEXT();
        else if (!strcmp(a,"--data"))    c->data     = NEXT();
        else if (!strcmp(a,"--block"))   c->block    = atoi(NEXT());
        else if (!strcmp(a,"--dim"))     c->dim      = atoi(NEXT());
        else if (!strcmp(a,"--ffn"))     c->ffn      = atoi(NEXT());
        else if (!strcmp(a,"--heads"))   c->heads    = atoi(NEXT());
        else if (!strcmp(a,"--layers"))  c->layers   = atoi(NEXT());
        else if (!strcmp(a,"--epochs"))  c->epochs   = atoi(NEXT());
        else if (!strcmp(a,"--lr"))      c->lr       = atof(NEXT());
        else if (!strcmp(a,"--wd"))      c->wd       = atof(NEXT());
        else if (!strcmp(a,"--gen"))     c->gen      = atoi(NEXT());
        else if (!strcmp(a,"--temp"))    c->temp     = atof(NEXT());
        else if (!strcmp(a,"--prompt"))  c->prompt   = NEXT();
        else if (!strcmp(a,"--maxseqs")) c->max_seqs = atoi(NEXT());
        else if (!strcmp(a,"--valfrac")) c->val_frac = atof(NEXT());
        else if (!strcmp(a,"--seed"))    c->seed     = (unsigned) atoi(NEXT());
        else if (!strcmp(a,"-h") || !strcmp(a,"--help")) { printf("%s",USAGE); exit(0); }
        else { fprintf(stderr,"Unknown option '%s'\n%s",a,USAGE); return 0; }
        #undef NEXT
    }
    if (strcmp(c->model,"transformer") && strcmp(c->model,"lstm")) {
        fprintf(stderr,"--model must be 'transformer' or 'lstm'\n");
        return 0;
    }
    if (!strcmp(c->model,"transformer") && (c->dim % c->heads) != 0) {
        fprintf(stderr,"--dim (%d) must be a multiple of --heads (%d)\n",
                c->dim,c->heads);
        return 0;
    }
    return 1;
}

typedef struct {
    unsigned char* text; /* raw bytes                          */
    long   n;            /* number of bytes                    */
    int    K;            /* vocabulary size                    */
    int    ch2idx[256];  /* byte -> class index, -1 if unused  */
    unsigned char idx2ch[256]; /* class index -> byte          */
} CORPUS;

/* Reads the whole file into memory. Returns 1 on success, 0 on failure. */
static int corpus_load(CORPUS* c, const char* path)
{
    FILE* f = fopen(path,"rb");
    if (f == NULL) {
        fprintf(stderr,"Cannot open '%s'.\n",path);
        fprintf(stderr,"Download TinyShakespeare from:\n"
          "  https://raw.githubusercontent.com/karpathy/char-rnn/"
          "master/data/tinyshakespeare/input.txt\n");
        return 0;
    }
    fseek(f,0,SEEK_END);
    long n = ftell(f);
    fseek(f,0,SEEK_SET);
    if (n < 2) {
        fprintf(stderr,"'%s' is too small.\n",path);
        fclose(f);
        return 0;
    }

    c->text = allocmem(n,1,unsigned char);
    long got = (long) fread(c->text,1,n,f);
    fclose(f);
    if (got != n) {
        fprintf(stderr,"Short read on '%s'.\n",path);
        return 0;
    }
    c->n = n;

    /* Build the vocabulary from the bytes actually present. */
    for (int i = 0; i < 256; i++)
        c->ch2idx[i] = -1;
    int K = 0;
    for (long i = 0; i < n; i++) {
        int b = c->text[i];
        if (c->ch2idx[b] < 0) { c->ch2idx[b] = K; c->idx2ch[K] = (unsigned char) b; K++; }
    }
    c->K = K;
    return 1;
}

static void corpus_free(CORPUS* c)
{
    freemem(c->text);
    c->text = NULL;
}

/* ------------------------------------------------------------------ */
/* Dataset construction                                                */
/*                                                                     */
/* Each sequence is T consecutive characters; its per-position target  */
/* is the next character. Sequences are cut with a stride chosen so     */
/* that, if the corpus allows more than max_seqs windows, they are      */
/* spread evenly across the whole text rather than all taken from the   */
/* front. Inputs and targets are stored as dense one-hot [n*T][K].      */
/* ------------------------------------------------------------------ */

typedef struct {
    fArr2D x;   /* [num*T][K] one-hot inputs   */
    fArr2D y;   /* [num*T][K] one-hot targets  */
    int*   len; /* [num] sequence lengths (all T) */
    int    num; /* number of sequences         */
    int    T;   /* sequence length             */
    int    K;   /* vocabulary size             */
} DATASET;

static void dataset_build(DATASET* d, const CORPUS* c, int T, int max_seqs)
{
    /* A window needs T inputs plus one more character for the last target. */
    long usable = c->n - 1;
    long max_windows = usable / T;               /* non-overlapping count   */
    if (max_windows < 1) { fprintf(stderr,"Corpus shorter than block.\n"); exit(1); }

    long stride = T;
    long num = max_windows;
    if (max_seqs > 0 && (long) max_seqs < max_windows) {
        num = max_seqs;
        stride = usable / num;                   /* spread across the text  */
        if (stride < 1) stride = 1;
    }

    int K = c->K;
    d->T = T; d->K = K; d->num = (int) num;
    d->x   = allocmem(num * T, K, float);
    d->y   = allocmem(num * T, K, float);
    d->len = allocmem(num, 1, int);

    typedef float (*ArrK)[K];
    ArrK X = (ArrK) d->x;
    ArrK Y = (ArrK) d->y;

    for (long s = 0; s < num; s++) {
        long start = s * stride;
        if (start + T + 1 > c->n) start = c->n - T - 1; /* clamp tail */
        for (int t = 0; t < T; t++) {
            long r  = s * T + t;
            int in  = c->ch2idx[c->text[start + t]];
            int tgt = c->ch2idx[c->text[start + t + 1]];
            X[r][in]  = 1.0f;
            Y[r][tgt] = 1.0f;
        }
        d->len[s] = T;
    }
}

static void dataset_free(DATASET* d)
{
    freemem(d->x); freemem(d->y); freemem(d->len);
    d->x = NULL; d->y = NULL; d->len = NULL;
}

/* ------------------------------------------------------------------ */
/* Model construction                                                  */
/* ------------------------------------------------------------------ */

static MODEL* build_model(const CONFIG* c, int K)
{
    const int T = c->block;
    const int D = c->dim;
    int is_xfmr = !strcmp(c->model,"transformer");

    /* Layer count: transformer = proj + N blocks + head; lstm = N + head. */
    int L = is_xfmr ? (c->layers + 2) : (c->layers + 1);

    /* batch size == sequence length T; one-hot input dim K; add bias; no norm */
    MODEL* m = model_create(L,T,K,1,0);

    if (is_xfmr) {
        model_add(m,dense_create(D,"none"),"dense");           /* embedding   */
        for (int i = 0; i < c->layers; i++)
            model_add(m,transformer_create(c->heads,T,D,c->ffn,0),"transformer");
    }
    else {
        for (int i = 0; i < c->layers; i++)
            model_add(m,lstm_create(D,0),"lstm");              /* stateful=0  */
    }

    model_add(m,dense_create(K,"Softmax"),"dense");            /* vocab head  */
    model_compile(m,"cross-entropy","adamw");
    return m;
}

/* ------------------------------------------------------------------ */
/* Sampling                                                            */
/* ------------------------------------------------------------------ */

/* Draws an index from a probability row with temperature. Because the head
 * is softmax, temperature on logits equals raising probabilities to 1/temp
 * and renormalising (softmax(log p / temp) is proportional to p^(1/temp)). */
static int sample_row(const float* p, int K, float temp)
{
    if (temp <= 0.0f) {                 /* greedy / argmax */
        int best = 0;
        for (int k = 1; k < K; k++) if (p[k] > p[best]) best = k;
        return best;
    }
    float w[256];
    float sum = 0.0f;
    float inv = 1.0f / temp;
    for (int k = 0; k < K; k++) {
        float pk = p[k] > 1e-16f ? p[k] : 1e-16f;
        w[k] = powf(pk,inv);
        sum += w[k];
    }
    float r = urand(0.0f,1.0f) * sum;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) { acc += w[k]; if (r <= acc) return k; }
    return K - 1;
}

/* ------------------------------------------------------------------ */
/* Autoregressive generation (fixed-width sliding window)              */
/* ------------------------------------------------------------------ */

static void generate(MODEL* m, const CORPUS* c, const CONFIG* cfg)
{
    const int T = cfg->block;
    const int K = c->K;

    /* Fill character used to pad an initially short context. */
    int fill = (c->ch2idx['\n'] >= 0) ? c->ch2idx['\n']
             : (c->ch2idx[' ']  >= 0) ? c->ch2idx[' '] : 0;

    int* window = allocmem(T,1,int);
    for (int t = 0; t < T; t++) window[t] = fill;

    /* Right-align the (mapped) prompt into the window and echo it. */
    const char* pr = cfg->prompt;
    int plen = (int) strlen(pr);
    printf("---- sample (%s, temp %.2f) ----\n",cfg->model,cfg->temp);
    for (int i = 0; i < plen; i++) {
        int idx = c->ch2idx[(unsigned char) pr[i]];
        if (idx < 0) continue;                     /* skip out-of-vocab */
        memmove(window,window + 1,(T - 1) * sizeof(int));
        window[T - 1] = idx;
        putchar(pr[i]);
    }

    fArr2D X = allocmem(T,K,float);
    fArr2D Y = allocmem(T,K,float);
    typedef float (*ArrK)[K];
    ArrK x = (ArrK) X;
    ArrK y = (ArrK) Y;

    for (int step = 0; step < cfg->gen; step++) {
        fltclr(X,T * K);
        for (int t = 0; t < T; t++) x[t][window[t]] = 1.0f;

        model_predict(m,X,Y,T);                    /* Y[t] = P(next | ctx<=t) */

        int next = sample_row(y[T - 1],K,cfg->temp);
        putchar(c->idx2ch[next]);

        memmove(window,window + 1,(T - 1) * sizeof(int));
        window[T - 1] = next;
    }
    printf("\n----------------------------------\n");

    freemem(X); freemem(Y); freemem(window);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
    CONFIG cfg;
    config_defaults(&cfg);
    if (!parse_args(argc,argv,&cfg)) return 1;

    unsigned int seed = cfg.seed;
    if (seed == 0) {
        struct timespec ts; clock_gettime(CLOCK_REALTIME,&ts);
        seed = (unsigned int)(ts.tv_sec ^ ts.tv_nsec);
    }
    init_lrng(seed);
    printf("seed %u\n",seed);

    /* Corpus + vocabulary */
    CORPUS corpus;
    if (!corpus_load(&corpus,cfg.data)) return 1;
    printf("corpus: %ld chars, vocab %d\n",corpus.n,corpus.K);

    /* Dataset of length-T windows */
    DATASET ds;
    dataset_build(&ds,&corpus,cfg.block,cfg.max_seqs);

    /* Train / validation split over whole sequences */
    int num_val = (int)(ds.num * cfg.val_frac);
    if (num_val < 0) num_val = 0;
    int num_tr  = ds.num - num_val;
    if (num_tr < 1) { num_tr = ds.num; num_val = 0; }

    fArr2D xTr = ds.x;
    fArr2D yTr = ds.y;
    int*   sTr = ds.len;

    /* Validation views start after the training sequences (contiguous rows). */
    typedef float (*ArrK)[ds.K];
    fArr2D xVd = num_val ? (fArr2D)(((ArrK) ds.x) + (size_t) num_tr * ds.T) : NULL;
    fArr2D yVd = num_val ? (fArr2D)(((ArrK) ds.y) + (size_t) num_tr * ds.T) : NULL;
    int*   sVd = num_val ? (ds.len + num_tr) : NULL;

    printf("model %s: %d layers, D %d, ffn %d, heads %d, block %d\n",
           cfg.model,cfg.layers,cfg.dim,cfg.ffn,cfg.heads,cfg.block);
    printf("sequences: %d train, %d val\n\n",num_tr,num_val);

    MODEL* m = build_model(&cfg,corpus.K);

    /* Train */
    float* losses = allocmem(cfg.epochs,1,float);
    float* accs   = allocmem(cfg.epochs,1,float);
    float* vloss  = num_val ? allocmem(cfg.epochs,1,float) : NULL;
    float* vaccs  = num_val ? allocmem(cfg.epochs,1,float) : NULL;

    model_fit(m,xTr,yTr,sTr,num_tr,
              xVd,yVd,sVd,num_val,
              cfg.epochs,cfg.lr,cfg.wd,
              losses,accs,vloss,vaccs,
              "final=1 verbose=2");

    printf("\n");

    /* Sample */
    generate(m,&corpus,&cfg);

    /* Cleanup */
    freemem(losses); freemem(accs); freemem(vloss); freemem(vaccs);
    model_free(m);
    dataset_free(&ds);
    corpus_free(&corpus);
    return 0;
}
