/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "array.h"
#include "sample.h"
#include "data.h"
#include "wav.h"
#include "ulaw.h"
#include "pcm.h"
#include "hann.h"
#include "lpc.h"
#include "lsp.h"
#include "filter.h"

SAMPLE samples[MAX_SAMPLES] = {0};
int samplecnt = 0;

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,"Syntax: feat2audio <feat infilename> <wav outfilename>\n");
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

    samplecnt = 0;
    for (int lcnt = 0;; lcnt++) {
        char buffer[20000];
        char *line = fgets(buffer,sizeof(buffer),fp);
        if (line != NULL) {
            char *phoneme = "phoneme";
            if (strncmp(line,"phoneme",strlen(phoneme)) == 0) // header
                continue;
            int err = parseline(line,lcnt,&samples[samplecnt]);
            if (err) {
                fprintf(stderr,"failed to read line %d from file '%s' - skipping\n",lcnt,featfilename);
                continue;
            }
            samplecnt++;
            if (samplecnt >= MAX_SAMPLES) {
                fprintf(stderr,"reached %d samples in file '%s' line %d - rest ignored\n",MAX_SAMPLES,featfilename,lcnt);
                break;
            }
        }
        else
            break;
    }
    fclose(fp);
    
    if (samplecnt == 0) {
        fprintf(stderr,"no vaild sample data in file '%s' - aborting\n",featfilename);
        return -1;
    }


    int audioSampleRate = 16000; // Somewhat arbitrary (divisble by 1000)

    // Calculate frame size (in milliseconds)   
    int frameSize = (int) (samples[0].duration * 1000 / samples[0].num_frames);

    // Calculate window size (in samples)
    int winSize = (2 * frameSize) * audioSampleRate / 1000;

    printf("winSize %d samples, frameSize %d millisec "
           "(first sample %5.3lf seconds)\n",
           winSize,frameSize,samples[0].duration);
 
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
        for (int j = 0; j < samples[i].num_frames; j++) {
            int lpcOrder = FRAME_SIZE - 2; // zcr, sigma
            // double zcr = samples[i].features[j][0]; // not used
            double sigma = exp(samples[i].features[j][1] * -30.0) - 1e-07;
            double lspCoeff[lpcOrder + 1];
            for (int k = 0; k < lpcOrder; k++)
                lspCoeff[k] = samples[i].features[j][k + 2];
            lspCoeff[lpcOrder] = 0.0;
            double lpcCoeff[lpcOrder + 1];
            lsp2lpc(lspCoeff,lpcCoeff,lpcOrder);
            LPCsynthesis(lpcCoeff,lpcOrder,sigma,winSize,window);
            for (int i = 0; i < winSize; i++)
                wrBuf[i] += window[i];
            writeWavFile(&wfout,wrBuf,winSize / 2);
            memmove(wrBuf,wrBuf + winSize / 2,(winSize / 2) * sizeof(wrBuf[0]));
            memset(wrBuf + winSize / 2,0,(winSize / 2) * sizeof(wrBuf[0]));
        }
    }
    closeWavFile(&wfout);
    return 0;
}
