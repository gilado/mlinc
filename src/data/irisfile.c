/* Copyright (c) 2023-2024 Gilad Odinak */
/* Read Iris dataset file               */
/* https://archive.ics.uci.edu/dataset/53/iris */
#include <stdio.h>
#include <string.h>
#include "irisfile.h"

const char* iris_class_names[IRIS_CLASS_CNT] = { 
    "setosa", "versicolor", "virginica" 
};

int read_iris_file(const char* irisfile, int numSamples,
                   float x[][IRIS_FEAT_CNT], int yc[])
{
    FILE *fp = fopen(irisfile,"rb");
    if (fp == NULL) {
        fprintf(stderr,"%s: failed to open file for read\n",irisfile);
        return 0;
    }
    for (int i = 0; i < numSamples; i++) {
        char buffer[256];
        char *line = fgets(buffer,sizeof(buffer),fp);
        if (line == NULL) {
            fprintf(stderr,"%s: at line %d: "
                           "failed to read from file\n",irisfile,i + 1);
            fclose(fp);
            return 0;
        }
        char cname[16];
        int cnt = sscanf(line,"%f,%f,%f,%f,%15s",
                         &x[i][0],&x[i][1],&x[i][2],&x[i][3],cname);
        if (cnt < 5) {
            fprintf(stderr,"%s: at line %d: "
                    "failed to parse 5 values from file\n",irisfile,i + 1);
            fclose(fp);
            return 0;
        }
        yc[i] = -1;
        for (int j = 0; j < IRIS_CLASS_CNT; j++) {
            if (strstr(cname,iris_class_names[j])) {
                yc[i] = j;
                break;
            }
        }
        if (yc[i] == -1) {
            fprintf(stderr,"%s: at line %d: i"
                    "unknown plant name %s\n",irisfile,i + 1,cname);
            fclose(fp);
            return 0;
        }
    }
    fclose(fp);
    return 1;
}
