/* Copyright (c) 2023-2024 Gilad Odinak */
/* Time measurment functions            */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <time.h>
#include "float.h"
#include "etime.h"

float current_time()
{
    struct timespec curtime;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &curtime);
    return (float) (curtime.tv_sec + (curtime.tv_nsec / 1000000000.0));
}

char* date_time(char buffer[20]) 
{
    time_t now;
    time(&now);
    struct tm *local = localtime(&now);
    strftime(buffer,20, "%Y-%m-%dT%H:%M:%S",local);
    return buffer;
}
