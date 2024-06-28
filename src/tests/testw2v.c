/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_WORDS 15000 // unique words
#define WDIM         63 // number of dimensions of word vectors
#define MAX_WORD_LEN 31 // maximum number of characters in a word

typedef struct word_embedding_s {
    char word[MAX_WORD_LEN + 1];
    int index;
    float embedding[WDIM];
    float norm;
} WORD_EMBEDDING;

static WORD_EMBEDDING word_embeddings[MAX_WORDS];
static int wordcnt = 0;

static int interactive = 0;

static float nrand(float mean, float stddev);
static float vecnorm(const float* x, int N);
static int parseline(char *line, int lcnt, WORD_EMBEDDING* word_embedding);
    
int main(int argc, char** argv)
{
    srand(time(NULL));
    if (argc >= 2 && strcmp(argv[1],"-i") == 0) {
        interactive = 1;
        if (argc > 2)
            argv[1] = argv[2];
        argc--;
    }
    if (argc < 2) {
        fprintf(stderr,"syntax: testembds [-i] <embeddings file>\n");
        return -1;
    }  
    const char* filename = argv[1];
    FILE* fp = fopen(filename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"could not open %s for read\n",filename);
        return -1;
    }
    for (wordcnt = 0; wordcnt < MAX_WORDS;) {
        char buffer[128 + WDIM * 16];
        char *txt = fgets(buffer,sizeof(buffer),fp);
        if (txt == NULL)
            break;
        if (strstr(txt,"word") && strstr(txt,"index")) // header
            continue; 
        int err = parseline(txt,wordcnt,&(word_embeddings[wordcnt]));
        if (err)
            break;
        wordcnt++;
    }
    printf("%d words loaded\n",wordcnt);
    if (wordcnt == 0)
        return 0;
    int quit = 0;
    for (int tests = 0; tests < 10;) {
        char *word = NULL;
        int idx = -1;
        if (interactive) {
            char buffer[MAX_WORD_LEN + 1];
            printf("Type a word, then press enter: ");
            char* txt = fgets(buffer,sizeof(buffer),stdin);
            if (txt != NULL) {
                while (*txt != '\0' && !isalnum(*txt))
                    txt++;
                char* end;
                for (end = txt; *end != '\0' && isalnum(*end); end++);
                *end = '\0';
            }
            if (txt == NULL || strlen(txt) == 0) {
                if (quit)
                    break;
                quit = 1;
                fprintf(stderr,"No valid input - try again, or press enter to quit\n");
                fflush(stderr);
                continue;
            }
            quit = 0;
            word = txt;
            printf("You entered '%s' => ",txt);
            fflush(stdout);
            for (idx = 0; idx < wordcnt; idx++)
                if (strcmp(word,word_embeddings[idx].word) == 0)
                    break;
            if (idx >= wordcnt) {
                fprintf(stderr,"\nWord not found in vocabulary, try again\n");
                continue;
            }
        }
        else {
            idx = (int) (rand() / ((float)RAND_MAX + 1) * wordcnt);
            word = word_embeddings[idx].word;
            printf("Radomly selected word '%s' => ",word);
            tests++;
        }
        // Create a vector similar to the selected word vector
        float simvec[WDIM];
        memcpy(simvec,word_embeddings[idx].embedding,WDIM * sizeof(float));
        float eps_dev = vecnorm(simvec,WDIM) * 0.2;
        for (int i = 0; i < WDIM; i++)
            simvec[i] += nrand(0,eps_dev);
        float simvec_norm = vecnorm(simvec,WDIM);

        float max_sim = -1.0;
        int max_idx = -1;
        for (int i = 0; i < wordcnt; i++) {
            float* tvec = word_embeddings[i].embedding;
            float tvec_norm =  word_embeddings[i].norm;
            // Calculate cosine similarity
            float tnum = 0.0;
            for (int j = 0; j < WDIM; j++)
                tnum += simvec[j] * tvec[j];
            float tden = simvec_norm * tvec_norm;
            float sim = tnum / tden;
            // Store the index of word with highest similiarity
            if (max_sim < sim) {
                max_sim = sim;
                max_idx = i;
            }
        }
        if (max_idx < 0 || max_idx >= wordcnt) {
            fprintf(stderr,"max_idx %d out of bound\n",max_idx);
            continue;
        }
        word = word_embeddings[max_idx].word;
        printf("%s (%f)\n",word,max_sim);
    }
}

/* Returns a random  from a normal distribution with the specified mean 
 * and standard deviation.
 */
static float nrand(float mean, float stddev) 
{
    float u1 = rand() / ((float)RAND_MAX + 1);
    float u2 = rand() / ((float)RAND_MAX + 1);
    // Box-Muller transform
    float z = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
    return mean + stddev * z; // Shift and scale
}

/* Calculates and returns the norm of the vector x.
 */
static float vecnorm(const float* x, int N)
{
    float sum = 0;
    for (int i = 0; i < N; i++) 
        sum += x[i] * x[i];
    return sqrt(sum);
}


static int parseline(char *line, int lcnt, WORD_EMBEDDING* word_embedding)
{
    char word[MAX_WORD_LEN+1];
    char *s = line;
    char *e = NULL;
    #define nexttok {if (*e == ',') e++; s = e;}
    
    // word
    e = index(s,',');
    if (e == NULL || e == s) {
        fprintf(stderr,"Malformed line %d: failed to read word\n",lcnt);
        return -1;
    }
    while (!isalnum(*s) && s != e) s++;
    int l = e - s;
    while (l >= 0 && !isalnum(s[l])) l--;
    l++;
    if (l <= 0 || l > MAX_WORD_LEN) {
        fprintf(stderr,"Malformed line %d: missing or invalid word\n",lcnt);
        return -1;
    }
    strncpy(word,s,l);
    word[l] = '\0';
    strcpy(word_embedding->word,word);
    //printf("'%s'\n",word_embedding->word);
    nexttok;

    // index
    int idx = (int) strtod(s,&e);        
    if (e == s) {
        fprintf(stderr,"Malformed line %d: failed to read word index\n",lcnt);
        return -1;
    }
    if (idx < 0) {
        fprintf(stderr,"Malformed line %d: invalid word index %d\n",lcnt,idx);
        return -1;
    } 
    word_embedding->index = idx;
    nexttok;
    
    // vector dimension
    int wdim = (int) strtod(s,&e);        
    if (e == s) {
        fprintf(stderr,"Malformed line %d: failed to read vector dimension\n",lcnt);
        return -1;
    }
    if (wdim != WDIM) {
        fprintf(stderr,"Malformed line %d: vector dimension is not %d\n",lcnt,WDIM);
        return -1;
    }
    nexttok;
    int i;
    float val[WDIM];
    for (i = 0; i < WDIM; i++) {
        val[i] = (float) strtod(s,&e);
        if (e == s) {
            fprintf(stderr,"Malformed line %d: failed to read vector value # %d\n",lcnt,i);
            break; // checked below: i <  WDIM
        }
        nexttok;
    }
    if (i <  WDIM) // failed to read a value
        return -1; 
    for (i = 0; i < WDIM; i++)
        word_embedding->embedding[i] = val[i];
    word_embedding->norm = vecnorm(val,WDIM);
    return 0;    
}



