/* Copyright (c) 2023-2024 Gilad Odinak */
#include <stdio.h>
#include <math.h>

#define NUM_CLASSES 3

double compute_loss(double output[NUM_CLASSES], double target[NUM_CLASSES]) {
    double loss = 0.0;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        loss += target[i] * log(output[i] + 1e-9);
    }
    
    return -loss;
}

int main() {
    double target[4][NUM_CLASSES] = {{0.0, 1.0, 0.0},{0.0, 1.0, 0.0},{0.0, 1.0, 0.0},{0.0, 1.0, 0.0}};
    double output[4][NUM_CLASSES] = {{0.0, 1.0, 0.0},{1.0, 0.0, 0.0},{0.2, 0.8, 0.0},{0.1, 0.8, 0.1}};

    for (int i = 0; i < 4; i++) {
        double loss = compute_loss(output[i], target[i]);
        printf("{%lf,%lf,%lf} {%lf,%lf,%lf}\n",
               output[i][0],output[i][1],output[i][2],
               target[i][0],target[i][1],target[i][2]);
        printf("Loss : %.4f\n", loss);
    }    
    return 0;
}
