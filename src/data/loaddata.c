/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <stdint.h>
#include <math.h>

#include "array.h"
#include "timitphn.h"
#include "sample.h"
#include "data.h"

int load_data(const char* listfile, SEQUENCE* sequences, int max_sequences)
{
    char *filenames[max_sequences];
    int filecnt = 0;
    void freeall(char* filenames[], int filecnt) {
        for (int i = 0; i < filecnt; i++) {
            free(filenames[i]);
        }
    }
    // Read file name list
    printf("Reading file list from %s\n",listfile);
    fflush(stdout);
    FILE* flp = fopen(listfile,"rb");
    if (flp == NULL) {
        fprintf(stderr,"%s: failed to open for read\n",listfile);
        return -1;
    }
    if (max_sequences <= 0)
        max_sequences = 0;
    else
    if (max_sequences > MAX_FILES)
        max_sequences = MAX_FILES;
    for (filecnt = 0; filecnt < max_sequences; filecnt++) { 
        char buffer[500];
        char *fn = fgets(buffer,sizeof(buffer),flp);
        if (fn == NULL) // End of file list
            break;
        buffer[sizeof(buffer) - 1] = '\0';
        char *eol = index(fn,'\n');
        if (eol != NULL) *eol = '\0';
        filenames[filecnt] = malloc(strlen(fn) + 1);
        if (filenames[filecnt] == NULL) {
            fprintf(stderr,"failed to allocate memory for filename %d\n",filecnt);
            fclose(flp);
            freeall(filenames,filecnt);
            return -1;            
        }
        strcpy(filenames[filecnt],fn);
    }
    // Load samples
    printf("Reading %d files\n",filecnt);
    fflush(stdout);

    int sequencecnt = 0;
    int allsamplecnt = 0;
    for (int i = 0; i < filecnt; i++) {
        fflush(stdout);
        FILE *fp = fopen(filenames[i],"rb");
        if (fp == NULL) {
            fprintf(stderr,"failed to open file %d '%s' - skipping\n",i,filenames[i]);
            continue;
        }
        SAMPLE samples[MAX_SAMPLES];
        int samplecnt = 0;
        for (int lcnt = 0;; lcnt++) {
            char buffer[20000];
            char *line = fgets(buffer,sizeof(buffer),fp);
            if (line != NULL) {
                char *phoneme = "phoneme";
                if (strncmp(line,"phoneme",strlen(phoneme)) == 0) // header
                    continue;
                int err = parseline(line,lcnt,&samples[samplecnt]);
                if (err) {
                    fprintf(stderr,"failed to read line %d from file %d '%s' - skipping\n",lcnt,i,filenames[i]);
                    continue;
                }
                samplecnt++;
                if (samplecnt >= MAX_SAMPLES) {
                    fprintf(stderr,"reached %d samples at file %d '%s' line %d - rest ignored\n",MAX_SAMPLES,i,filenames[i],lcnt);
                    break;
                }
            }
            else
                break;
        }
        fclose(fp);
        if (samplecnt == 0) {
            fprintf(stderr,"file %d '%s' contained no data - skipping\n",i,filenames[i]);
            continue;
        }
        int size = samplecnt * sizeof(SAMPLE);
        sequences[sequencecnt].samples = (SAMPLE *) malloc(size);
        if (sequences[sequencecnt].samples == NULL) {
            fprintf(stderr,"file %d '%s': failed to allocate %d bytes for %d samples - rest ignored\n",i,filenames[i],size,samplecnt);
            break;
        }
        memcpy(sequences[sequencecnt].samples,samples,size);
        sequences[sequencecnt].num_samples = samplecnt;
        sequencecnt++;
        if (sequencecnt >= MAX_FILES) {
            fprintf(stderr,"reached %d sequences at file %d '%s' - rest ignored\n",sequencecnt,i,filenames[i]);
            break;
        }
        allsamplecnt += samplecnt;
    }
    printf("Read %d files, loaded %d sequences, %d samples\n",
                                             filecnt,sequencecnt,allsamplecnt);
    freeall(filenames,filecnt);
    return sequencecnt;
}


int parseline(char *line, int lcnt, SAMPLE* restrict sample)
{
    char phn[SIZEPHN];
    char *s = line;
    char *e = NULL;
    #define nexttok {if (*e == ',') e++; s = e;}
    
    sample->id = lcnt;

    e = index(s,',');
    if (e == NULL || e == s) {
        fprintf(stderr,"Malformed line %d: failed to read phoneme\n",lcnt);
        return -1;
    }
    int l = e - s;

    // phoneme
    strncpy(phn,s,SIZEPHN - 1);
    if (l > 0 && l < SIZEPHN)
       while (phn[--l] == ' ');
    if (l < 0 || l >= SIZEPHN) {
        fprintf(stderr,"Malformed line %d: missing or invalid phoneme '%s'\n",lcnt,phn);
        return -1;
    }
    phn[++l] = '\0';
    nexttok;

    // label
    int pos = (int) strtod(s,&e);        
    if (e == s) {
        fprintf(stderr,"Malformed line %d: failed to read label\n",lcnt);
        return -1;
    }    
    for (int i = 0; i < NUM_CLASSES; i++)
        sample->expected_output[i] = 0.0;
    sample->expected_output[pos] = 1.0;
    nexttok;

    // start time
    float stime = (float) strtod(s,&e);
    if (e == s) {
        fprintf(stderr,"Malformed line %d: failed to read start time\n",lcnt);
        return -1;
    }
    nexttok;

    // end time
    float etime = (float) strtod(s,&e);
    if (e == s) {
        fprintf(stderr,"Malformed line %d: failed to read end time\n",lcnt);
        return -1;
    }
    nexttok;

    // duration
    sample->duration = (float) (etime - stime);

    // file name
    e = index(s,',');
    if (e == NULL) {
        fprintf(stderr,"Malformed line %d: failed to skip file name\n",lcnt);
        return -1;
    }
    nexttok;
    
    // frame size
    int frame_size = (int) strtod(s,&e);        
    if (e == s) {
        fprintf(stderr,"Malformed line %d: failed to read frame size\n",lcnt);
        return -1;
    }
    if (frame_size != FRAME_SIZE) {
        fprintf(stderr,"Malformed line %d: frame size is not %d\n",lcnt,FRAME_SIZE);
        return -1;
    }
    nexttok;
    // number of vectors
    sample->num_frames = (int) strtod(s,&e);        
    if (e == s) {
        fprintf(stderr,"Malformed line %d: failed to read number of frames\n",lcnt);
        return -1;
    }
    if (sample->num_frames > MAX_FRAMES) {
        fprintf(stderr,"Malformed line %d: too many frames: %d\n",lcnt,sample->num_frames);
        return -1;
    }
    if (sample->num_frames == 0) {
        fprintf(stderr,"Malformed line %d: no frames\n",lcnt);
        return -1;
    }
    nexttok;

    double feat[MAX_FRAMES * FRAME_SIZE];
    int fcnt = sample->num_frames * FRAME_SIZE;
    int fno = 0;
    for (fno = 0; fno < fcnt; fno++) {
        feat[fno] = strtod(s,&e);
        if (e == s) {
            fprintf(stderr,"Malformed line %d: failed to read feature %d\n",lcnt,fno);
            break; // checked below: fno <  fcnt
        }
        nexttok;
    }
    if (fno <  fcnt) // failed to read feature
        return -1;
    
    for (int i = 0, fno = 0; i < sample->num_frames; i++) {
        for (int j = 0; j < FRAME_SIZE; j++)
            sample->features[i][j] = feat[fno++];
    }
            
    // pad to complete a segment
    for (int i = sample->num_frames; i < MAX_FRAMES; i++)
        for (int j = 0; j < FRAME_SIZE; j++) 
            sample->features[i][j] = 0.0;
    
    return 0;    
}



