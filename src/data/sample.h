#ifndef _SAMPLE_H
#define _SAMPLE_H

#define FRAME_SIZE  14
#define MAX_FRAMES  32
#define NUM_CLASSES 64

typedef struct sample_s {
    int id;
    double duration; // seconds
    int num_frames; // input[0 ... num_frames-1]
    float features[MAX_FRAMES][FRAME_SIZE];
    float expected_output[NUM_CLASSES];
} SAMPLE;

#endif
