/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "wav.h"
#include "sphere.h"
#include "ulaw.h"
#include "pcm.h"

int main(int argc, char **argv)
{
    char* infilename = argv[1];
    char* outfilename = argv[2];
    SPHFILE sfin, *sfp;
    WAVFILE wfout, *wfp;

    if (argc < 3) {
        fprintf(stderr,"Syntax: test <infilename> <outfilename>\n");
        return 0;
    }
    if (strcmp(argv[1],argv[2]) == 0) {
        fprintf(stderr,"Input and output file names must be different\n");
        return 0;
    }
    sfp = openSphereFile(infilename,"r",&sfin);
    if (sfp == NULL) 
        return 0;
    
    wfout.audioFormat = 3; // float
    wfout.sampleRate = sfp->sampleRate; // same as input
    wfout.bitDepth = 32;
    wfout.numChannels = 1;  
    wfp = openWavFile(outfilename,"w",&wfout);
    if (wfp == NULL) {
        closeSphereFile(&sfin);
        return 0;
    }
    int sampleRate = sfin.sampleRate;
    size_t bufSize = (sampleRate == 8000) ? 160 : 320;
    float rdBuf[bufSize];
    float wrBuf[bufSize];

    memset(rdBuf,0,bufSize * sizeof(rdBuf[0]));
    memset(wrBuf,0,bufSize * sizeof(wrBuf[0]));

    for (size_t cnt = 0; cnt < sfin.numSamples; cnt += bufSize) {
        float fltBuf[bufSize];
        size_t samplesRead = 0;
        if (sfin.audioFormat != 3) {
            int16_t pcmBuf[bufSize];
            if (sfin.audioFormat == 7) {
                uint8_t ulawBuf[bufSize];
                samplesRead = readSphereFile(&sfin,ulawBuf,bufSize);
                ulaw2pcm(ulawBuf,pcmBuf,samplesRead);
            }
            else
                samplesRead = readSphereFile(&sfin,pcmBuf,bufSize);
            if (sfin.endianess == 'b') {
                for (int i = 0; i < ((int) bufSize); i++)
                    pcmBuf[i] = __builtin_bswap16(pcmBuf[i]);
            } 
            pcm2flt(pcmBuf,fltBuf,samplesRead);
        }
        else {
            samplesRead = readSphereFile(&sfin,fltBuf,bufSize);
            if (sfp->endianess == 'b') {
                for (int i = 0; i < ((int) bufSize); i++)
                    fltBuf[i] = __builtin_bswap32(fltBuf[i]);
            }
        } 

        if (samplesRead < bufSize) {
            memset(fltBuf + samplesRead,0,
                     (bufSize - samplesRead) * sizeof(rdBuf[0]));
        }
        for (int i = 0; i < ((int) bufSize); i++)
            rdBuf[i] = fltBuf[i];

        for (size_t i = 0; i < bufSize; i++)
            wrBuf[i] = rdBuf[i];

        size_t samplesWritten = writeWavFile(&wfout,wrBuf,bufSize);

        if (samplesWritten < bufSize) {
            fprintf(stderr,"Failed to write to output file - aborting\n");
            break;
        }
    }
    closeSphereFile(&sfin);    
    closeWavFile(&wfout);    
}
