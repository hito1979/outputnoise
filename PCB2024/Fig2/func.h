// functions.h
#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#define MAX_LINE_SIZE 1024

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "func.h"  // Include the header for function declarations

// Global variable declarations


// Function declarations
void CV(double a, double b, double *A, double *B, double k, int step, double W, double dt, double e, double D, int N, int count, int start_time, int distance, int sample_period, int repetition, int base_seed, double *CV_x);
double gauss(double uniform1, double uniform2);
double f(double theta, double *A, double *B, int N);
void measure_period(int ids_peak[], int size, int sample_period, double dt, double *period, int distance);
void normalise_vector(double *x, int dim);
void writedata_2d(char *file, char **header, double **data, int rows, int columns);
void readdata(const char *file, double *dataArray);

// Add other function declarations...

#endif // FUNCTIONS_H



