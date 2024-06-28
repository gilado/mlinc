/* Copyright (c) 2023-2024 Gilad Odinak */
/* Time measurment functions            */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <time.h>
#include "etime.h"

float current_time()
{
    struct timespec currentTime;
    clock_gettime(CLOCK_MONOTONIC, &currentTime);
    return ((float) currentTime.tv_sec) + 
           ((float) currentTime.tv_nsec) / 1000000000;
}

char* date_time(char buffer[20]) 
{
    time_t now;
    time(&now);
    struct tm *local = localtime(&now);
    strftime(buffer,20, "%Y-%m-%dT%H:%M:%S",local);
    return buffer;
}
