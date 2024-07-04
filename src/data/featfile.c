/* Copyright (c) 2023-2024 Gilad Odinak */
/* Read speech feature files                 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include "float.h"
#include "delta.h"
#include "featfile.h"

/* Reads speech samples from speech feature file.
 *
 * The file contains a sequence of phonemes, one phoneme per text line.
 * Each phoneme is represented by a number of vectors (aka frames) and 
 * each frame consists of a vector of real numbers; all the vectors of
 * a phoneme have the same label, the phonme's name.
 *
 * The feature vectors read from the files are expanded to include
 * deltas and delta-deltas of their values.
 *
 * Parameters
 *   fp   - A pointer to a feature file open for read.
 *   maxs - The maximum number of samples (number of rows in x,
 *          and number of elements in yc)
 *   
 * Returns:
 *   An array of feature vectors in x, their corresponding labels in yc.
 *   
 *   This function returns the actual number of samples.
 *   That also is the returned number of rows in x, elements in yc.
 *   Returns 0 if an error occured.
 *
 * Notes:
 *   The set of phonemes can be CMU 39 phonemes, or TIMIT 61 phonemes.
 *   In the later case TIMIT phonemes are mapped to their CMU equivalents.
 *   Reference: Speaker-Independent Phone Recognition, Lee and Hon 1989
 */
int read_feature_file(FILE* fp, int maxs, 
                      float x[][EXPENDED_FEAT_CNT], int yc[])
{
    int lineno = 0;
    int vecinx = 0;
    int seqlen = 0;
    while (vecinx < maxs) {
        char buf[20000];
        char* line = fgets(buf,sizeof(buf),fp);
        if (line == NULL) /* End of file */
            break;
        lineno++;
        int len = strlen(line);
        if (len == 0 || line[len - 1] != '\n') {
            fprintf(stderr,"Line %d too long or malformed\n",lineno);
            return 0;
        }
        /* Remove all whitespace in current line */
        len = 0;
        for (int i = 0; line[i] != '\0'; i++)
            if (!isspace(line[i]))
                line[len++] = line[i];
        line[len] = '\0';
        const char* header = "phoneme,"; /* Header line starts with this */
        if (strncmp(line,header,strlen(header)) == 0) /* Header */
            continue;
        /* Replace commas with spaces */
        for (int i = 0; line[i] != '\0'; i++)
            if (line[i] == ',')
                line[i] = ' ';
        /* Read the first 7 fields */
        const char* fmt = "%4s %d " FMTF " " FMTF " %255s %d %d";
        char ph[5];
        int label;
        float stime,etime;
        char fn[256];
        int fcnt, nfrm;             
        int cnt = sscanf(line,fmt,ph,&label,&stime,&etime,fn,&fcnt,&nfrm);
        if (cnt < 7) {
            fprintf(stderr,"Line %d is malformed\n",lineno);
            return 0;
        }
        if (fcnt != FEAT_CNT) {
            fprintf(stderr,"In line %d: feature count (fcnt) is %d, "
                                        "should be %d\n",lineno,fcnt,FEAT_CNT);
            return 0;
        }
        if (nfrm == 0) /* Line has no features */
            continue;
        /* Advance line to space past last scanned value */
        while (cnt > 0 && *line != '\0')
            if (*line++ == ' ')
                cnt--;
        /* Read all vectors of current phoneme */
        for (int i = 0; i < nfrm; i++) {
            for (int j = 0; j < FEAT_CNT; j++) {
                char* end = line;
                float feat;
                feat = strtof(line,&end);
                if (line == end) {
                    fprintf(stderr,"In line %d: malformed feature #%d\n",
                                                      lineno,i * FEAT_CNT + j);
                    return 0;
                 }
                 x[vecinx][j] = feat; 
                 line = end;
            }
            (void) timit_phoneme_names; /* Acquiesce the complier */
            (void) reduced_phoneme_names;
            yc[vecinx] = timit2reduced[label]; /* TIMIT -> CMT    */
            if (i == nfrm - 1)          /* Last vector of phoneme */
                yc[vecinx] += EOP;      /* Mark as end of phoneme */
            vecinx++;
            if (vecinx >= maxs) {
                fprintf(stderr,"In line %d: reached %d samples, "
                                            "ignoring the rest\n",lineno,maxs);
                break;
             }
        }
        seqlen += nfrm;
    }
    /* Expand features */
    int M = seqlen;
    int N = EXPENDED_FEAT_CNT;
    calculate_deltas(x[vecinx - M],M,N,0,14,14,3);  /* deltas       */
    calculate_deltas(x[vecinx - M],M,N,14,28,14,3); /* delta-deltas */
    calculate_deltas(x[vecinx - M],M,N,0,14,42,5);  /* deltas       */
    calculate_deltas(x[vecinx - M],M,N,42,56,14,5); /* delta-deltas */
    return seqlen;
}


/* Reads speech samples from speech feature files.
 *
 * Each file contains a sequence of phonemes, one phoneme per text line.
 * Each phoneme is represented by a number of vectors (aka frames) and 
 * each frame consists of a vector of real numbers; all the vectors of
 * a phoneme have the same label, the phonme's name.
 *
 * The feature vectors read from the files are expanded to include
 * deltas and delta-deltas of their values.
 *
 * Parameters
 * - input_dir is a directory containing speech feature files, whose name
 *   ends with '.feat'.
 * - file_list is a text file containing the names of the files to be loaded.
 * - max_sequences is the maximum number of sequences (size of seq_length[])
 * - max_samples is the maximum number of samples (number of rows in x, and
 *   number of elements in yc)
 *   
 * Returns:
 *   An array of feature vectors in x, their corresponding labels in yc,
 *   and the length of each sequence of feature vectors in seq_length.
 *   
 *   This function returns the actual number of sequences.
 *   That also is the number of elements in seq_length.
 *   Returns 0 if an error occured.
 *
 *   The actual number of samples returned can be calculated by summing
 *   all values in seq_length.
 *
 * Notes:
 *   The set of phonemes can be CMU 39 phonemes, or TIMIT 61 phonemes.
 *   In the later case TIMIT phonemes are mapped to their CMU equivalents.
 *   Reference: Speaker-Independent Phone Recognition, Lee and Hon 1989
 */
int read_feature_files(const char* input_dir, const char* file_list,
                       int max_sequences, int *seq_length,
                       int max_samples, float x[][EXPENDED_FEAT_CNT], int yc[])
{
    const int maxpath = 512;
    char buffer[3 * maxpath];
    if ((int) strlen(input_dir) >= maxpath) {
        fprintf(stderr,"Directory name too long: '%s'\n",input_dir);
        return 0;
    }
    FILE *lfp = fopen(file_list,"rb");
    if (lfp == NULL) {
        fprintf(stderr,"Failed to open '%s' for read\n",file_list);
        return 0;
    }
    
    int vecinx = 0;
    int seqinx = 0;
    int fileno = 0;
    /* Iterate over all files */
    while (seqinx < max_sequences && vecinx < max_samples) {
        strcpy(buffer,input_dir);
        int pfxlen = strlen(buffer);
        if (buffer[pfxlen - 1] != '/')
            strcat(buffer,"/");
        char* filepath = buffer + strlen(buffer);
        filepath = fgets(filepath,maxpath,lfp);
        if (filepath == NULL || strlen(filepath) == 0)
            break;      /* End of file list */
        for (int i = 0; filepath[i] != '\0'; i++)
            if (filepath[i] == '/')
                filepath[i] = '_';
        filepath = buffer;    /* filepath now points to file path */
        filepath[strlen(filepath) - 1] = '\0'; /* Punch out end of line char */
        char *ext = rindex(filepath,'.');
        if (ext != NULL)
            *ext = '\0'; /* Remove extension if any */
        strcat(filepath,".FEAT");  
        fileno++;
        FILE* fp = fopen(filepath,"rb");
        if (fp == NULL) {
            fprintf(stderr,"Failed to open file '%s' (%d) for read - "
                                            "skipping file\n",filepath,fileno);
            continue;
        }
        int veccnt = read_feature_file(fp,max_samples - vecinx, 
                                                       x + vecinx,yc + vecinx);
        fclose(fp);
        seq_length[seqinx++] = veccnt;
        vecinx += veccnt;
    }
    fclose(lfp);
    return seqinx;
}

const char* timit_phoneme_names[TIMIT_PHONEME_CNT] = {
    "","aa","ae","ah","ao","aw","ax","axr",
    "ax-h","ay","b","bcl","ch","d","dcl","dh",
    "dx","eh","el","em","en","eng","er","ey",
    "f","g","gcl","h","hh","hv","ih","ix",
    "iy","jh","k","kcl","l","m","n","ng",
    "nx","ow","oy","p","pcl","q","r","s",
    "sh","t","tcl","th","uh","uw","ux","v",
    "w","wh","y","z","zh","pau","epi","h#"
};

const char* reduced_phoneme_names[REDUCED_PHONEME_CNT] = {
    "sil","aa","ae","ah","aw","ay","b","ch",
    "d","dh","dx","eh","er","ey","f","g",
    "hh","ih","iy","jh","k","l","m","n",
    "ng","ow","oy","p","r","s","sh","t",
    "th","uh","uw","v","w","y","z"
};

const int timit2reduced[TIMIT_PHONEME_CNT] = {
    0,/*""->sil*/   1,/*aa->aa*/    2,/*ae->ae*/    3,/*ah->ah*/
    1,/*ao->aa*/    4,/*aw->aw*/    3,/*ax->ah*/   12,/*axr->er*/
    3,/*ax-h->ah*/  5,/*ay->ay*/    6,/*b->b*/      0,/*bcl->sil*/
    7,/*ch->ch*/    8,/*d->d*/      0,/*dcl->sil*/  9,/*dh->dh*/
   10,/*dx->dx*/   11,/*eh->eh*/   21,/*el->l*/    22,/*em->m*/
   23,/*en->n*/    24,/*eng->ng*/  12,/*er->er*/   13,/*ey->ey*/
   14,/*f->f*/     15,/*g->g*/      0,/*gcl->sil*/ 16,/*h->hh */
   16,/*hh->hh*/   16,/*hv->hh*/   17,/*ih->ih*/   17,/*ix->ih*/
   18,/*iy->iy*/   19,/*jh->jh*/   20,/*k->k*/      0,/*kcl->sil*/
   21,/*l->l*/     22,/*m->m*/     23,/*n->n*/     24,/*ng->ng*/
   23,/*nx->n*/    25,/*ow->ow*/   26,/*oy->oy*/   27,/*p->p*/
    0,/*pcl->sil*/  0,/*q->sil*/   28,/*r->r*/     29,/*s->s*/
   30,/*sh->sh*/   31,/*t->t*/      0,/*tcl->sil*/ 32,/*th->th*/
   33,/*uh->uh*/   34,/*uw->uw*/   34,/*ux->uw*/   35,/*v->v*/
   36,/*w->w*/     36,/*wh->w*/    37,/*y->y*/     38,/*z->z*/
   30,/*zh->sh*/    0,/*pau->sil*/  0,/*epi->sil*/  0/*h#->sil*/
};
