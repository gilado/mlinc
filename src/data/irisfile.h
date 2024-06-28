/* Copyright (c) 2023-2024 Gilad Odinak */
/* Read Iris dataset file               */
#ifndef IRISFILE_H
#define IRISFILE_H

#define IRIS_SAMPLE_CNT 150
#define IRIS_FEAT_CNT     4
#define IRIS_CLASS_CNT    3

extern const char* iris_class_names[IRIS_CLASS_CNT];
/* = { "setosa", "versicolor", "virginica" } */

int read_iris_file(const char* irisfile, int numSamples,
                   float x[][IRIS_FEAT_CNT], int yc[]);

#endif
