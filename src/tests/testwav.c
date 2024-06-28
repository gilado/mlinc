/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "wav.h"
#include "ulaw.h"
#include "pcm.h"

int main(int argc, char **argv)
{
  char* infilename = argv[1];
  char* outfilename = argv[2];
  WAVFILE wfin, wfout, *wfp;
  char *datain = NULL;
  char *dataout = NULL;
  size_t inSamples, outSamples;

  if (argc < 3) {
    fprintf(stderr,"syntax: test <infilename> <outfilename>\n");
    return 0;
  }
  if (strcmp(argv[1],argv[2]) == 0) {
    fprintf(stderr,"input and output file names must be different\n");
    return 0;
  }
  wfp = openWavFile(infilename,"r",&wfin);
  if (wfp == NULL) 
    return 0;
  datain = malloc(wfp->dataSize);
  if (datain == NULL) {
    fprintf(stderr,"failed to allocate read buffer of %d bytes\n",wfp->dataSize);
    closeWavFile(wfp);    
    return 0;
  }
  // Read entire data
  inSamples = readWavFile(wfp,datain,wfp->numSamples);
  if (inSamples != wfp->numSamples) {
    fprintf(stderr,"failed to read all samples (read %ld of %d)\n",inSamples,wfp->numSamples);
    free(datain);
    closeWavFile(wfp);    
    return 0;
  }
  closeWavFile(wfp);    
/*
  // Write PCM as uLaw
  size_t dosize = wfp->dataSize / 2;
  dataout = malloc(dosize);
  if (dataout == NULL) {
    fprintf(stderr,"failed to allocate write buffer of %d bytes\n",dosize);
    free(datain);
    return 0;
  }

  pcm2ulaw((const int16_t*) datain, (uint8_t*) dataout,inSamples);
  
  wfout.audioFormat = 7;
  wfout.sampleRate = wfp->sampleRate;
  wfout.bitDepth = 8;
  wfout.numChannels = wfp->numChannels;
*/

/*
  // Write uLAw as PCM
  size_t dosize = wfp->dataSize * 2;
  dataout = malloc(dosize);
  if (dataout == NULL) {
    fprintf(stderr,"failed to allocate write buffer of %d bytes\n",dosize);
    free(datain);
    return 0;
  }

  ulaw2pcm((const uint8_t*) datain, (int16_t*) dataout,inSamples);
  
  wfout.audioFormat = 1;
  wfout.sampleRate = wfp->sampleRate;
  wfout.bitDepth = 16;
  wfout.numChannels = wfp->numChannels;
*/

  // Write PCM as float
  size_t dosize = wfp->dataSize * 2;
  dataout = malloc(dosize);
  if (dataout == NULL) {
    fprintf(stderr,"failed to allocate write buffer of %ld bytes\n",dosize);
    free(datain);
    return 0;
  }

  pcm2flt((const int16_t*) datain, (float*) dataout,inSamples);
  
  wfout.audioFormat = 3;
  wfout.sampleRate = wfp->sampleRate;
  wfout.bitDepth = 32;
  wfout.numChannels = wfp->numChannels;
  
  wfp = openWavFile(outfilename,"w",&wfout);
  if (wfp == NULL) {
    free(datain);
    free(dataout);
    return 0;
  }
  
  // Write entire data
  outSamples = writeWavFile(wfp,dataout,inSamples);
  if (outSamples != inSamples)
    fprintf(stderr,"failed to write all samples (wrote %ld of %ld)\n",outSamples,inSamples);

  free(datain);
  free(dataout);
  closeWavFile(wfp);    
}
