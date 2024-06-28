/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*#include "hyperparams.h"*/
#include "model.h"
/*#include "modelmem.h"*/
#include "timitphn.h"
#include "sample.h"
#include "data.h"
/*#include "train.h"*/

#define MAX_SAMPLES 10000
SAMPLE samples[MAX_SAMPLES] = {0};
int samplecnt = 0;

int main(int argc, char **argv)
{
    double loss = 0.0;
    double accuracy = 0.0;
    if (argc < 3) {
        fprintf(stderr,"Syntax: timittest <model file> <feat file>\n");
        return 0;
    }
    if (strcmp(argv[1],argv[2]) == 0) {
        fprintf(stderr,"model file and features file names must be different\n");
        return 0;
    }

    char* modelfilename = argv[1];
    int err = loadmodel(modelfilename,&xfmr);
    if (err < 0) {
        fprintf(stderr,"failed to load model file (missing or corrupt)\n");
        return -1;
    }
    
    char* featfilename = argv[2];
    printf("Openning '%s' for read.\n",featfilename);
    FILE* fp = fopen(featfilename,"rb");
    if (fp == NULL) {
        fprintf(stderr,"%s: failed to open for read\n",featfilename);
        return 0;
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
    test(&xfmr,samplecnt,samples,&loss,&accuracy);
    printf("tested on %d samples, loss %.2lf, accuracy %.2lf\n",samplecnt,loss,accuracy);
}
