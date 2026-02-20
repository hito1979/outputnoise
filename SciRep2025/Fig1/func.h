// functions.h
#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#define MAX_LINE_SIZE 1024

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Global variable declarations


// Function declarations
void CV(double a, double b, double scale, double shift, int num_pow,  double T, double k_u, double k_v, double k_w, double k_x, int step, double dt, double e, double D, int count, int start_time, int distance, int sample_period, int repetetion, int base_seed, double *CV_w, double *CV_x);
double gauss(double uniform1, double uniform2);
void measure_period(int ids_peak[], int size, int sample_period, double dt, double *period, int distance);

#endif // FUNCTIONS_H



