/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <ctype.h>
#include <math.h>
#include "array.h"
#include "wav.h"
#include "ulaw.h"
#include "pcm.h"
#include "hann.h"
#include "lpc.h"
#include "lsp.h"
#include "filter.h"

#define TIMIT_FEAT_CNT      14
#define MAX_SAMPLES 1000 /* 10 seconds */
float samples[MAX_SAMPLES][TIMIT_FEAT_CNT];
int samplecnt = 0;

/* Somewhat arbitrary (divisble by 1000) */
int audioSampleRate = 16000;
/* These are calculated from first phoneme data */
float duration = 0; // in seconds
int frameSize = 0;  // in milliseconds
int winSize = 0;    // in samples
int featcnt = 0;    // sample vector dimension

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,
                "Syntax: feat2audio <feat infilename> <wav outfilename>\n");
        return 0;
    }
    WAVFILE wfout, *wfp;
    char* featfilename = argv[1];
    char* wavfilename = argv[2];

    if (strcmp(featfilename,wavfilename) == 0) {
        fprintf(stderr,"Input and output file names must be different\n");
        return -1;
    }

    FILE *fp = fopen(featfilename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"Failed to open '%s' for read\n",featfilename);
        return -1;
    }

    int lineno = 0;
    int vecinx = 0;
    for (;;) {
        char buffer[20000];
        char *line = fgets(buffer,sizeof(buffer),fp);
        if (line == NULL) /* End of file */
            break;
        lineno++;
        int len = strlen(line);
        if (len == 0 || line[len - 1] != '\n') {
            fprintf(stderr,"In file '%s' line %d: line too long "
                                          "or malformed\n",featfilename,lineno);
            fclose(fp);
            return -1;
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
            fprintf(stderr,"In file '%s' line %d: malformed line\n",
                                                          featfilename,lineno);
            fclose(fp);
            return -1;
        }
        if (fcnt != TIMIT_FEAT_CNT) {
            fprintf(stderr,"In file '%s' line %d: fcnt is %d, should be %d\n",
                                      featfilename,lineno,fcnt,TIMIT_FEAT_CNT);
            fclose(fp);
            return -1;
        }
        if (nfrm == 0) /* Line has no features */
            continue;

        if (winSize == 0) {
            duration = etime - stime; // in seconds
            frameSize = (int) duration * 1000 / nfrm; // in milliseconds
            winSize = (2 * frameSize) * audioSampleRate / 1000; // in samples
            featcnt = fcnt;
        }
        /* Advance line to space past last scanned value */
        while (cnt > 0 && *line != '\0')
            if (*line++ == ' ')
                cnt--;
        /* Read all vectors of current phoneme */
        for (int i = 0; i < nfrm; i++) {
            for (int j = 0; j < TIMIT_FEAT_CNT; j++) {
                char* end = line;
                float feat;
                feat = strtof(line,&end);
                if (line == end) {
                    fprintf(stderr,
                            "In file '%s' line %d: malformed feature #%d\n",
                            featfilename,lineno,i * TIMIT_FEAT_CNT + j);
                    fclose(fp);
                    return -1;
                 }
                 samples[vecinx][j] = feat; 
                 line = end;
            }
            vecinx++;
            samplecnt = vecinx;
            if (vecinx >= MAX_SAMPLES) {
                fprintf(stderr,
                        "In file '%s' line %d: reached %d samples, "
                        "ignoring the rest\n",featfilename,lineno,MAX_SAMPLES);
                break;
            }
        }
    }
    fclose(fp);
    
    if (samplecnt == 0) {
        fprintf(stderr,
                "no vaild sample data in file '%s' - aborting\n",featfilename);
        return -1;
    }

    printf("winSize %d samples, frameSize %d millisec "
           "(first sample %5.3lf seconds)\n",
           winSize,frameSize,duration);
 
    wfout.audioFormat = 3; // float
    wfout.sampleRate = audioSampleRate;
    wfout.bitDepth = 32;
    wfout.numChannels = 1;  
    wfp = openWavFile(wavfilename,"w",&wfout);
    if (wfp == NULL)
        return -1;

    float window[winSize];
    float wrBuf[winSize];

    memset(wrBuf,0,winSize * sizeof(wrBuf[0]));

    for (int i = 0; i < samplecnt; i++) {
        // double zcr = samples[i][0]; // not used
        double sigma = exp(samples[i][1] * -30.0) - 1e-07;
        int lpcOrder = featcnt - 2; // zcr, sigma
        double lspCoeff[lpcOrder + 1];
        for (int k = 0; k < lpcOrder; k++)
            lspCoeff[k] = samples[i][k + 2];
        lspCoeff[lpcOrder] = 0.0;
        double lpcCoeff[lpcOrder + 1];
        lsp2lpc(lspCoeff,lpcCoeff,lpcOrder);
        LPCsynthesis(lpcCoeff,lpcOrder,sigma,winSize,window);
        for (int j = 0; j < winSize; j++)
            wrBuf[j] += window[j];
        writeWavFile(&wfout,wrBuf,winSize / 2);
        memmove(wrBuf,wrBuf + winSize / 2,(winSize / 2) * sizeof(wrBuf[0]));
        memset(wrBuf + winSize / 2,0,(winSize / 2) * sizeof(wrBuf[0]));
    }
    closeWavFile(&wfout);
    return 0;
}
