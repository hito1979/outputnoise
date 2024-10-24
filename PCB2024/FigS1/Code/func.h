// functions.h
#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#define MAX_LINE_SIZE 1024

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>


// Function declarations
void CV_u(double a, double b, double scale, double shift, double num_pow,  double T, double k_u, double k_v, double k_w, double k_x, int step, double dt, double e, double D, int count, int start_time, int distance, int sample_period, int repetetion, int base_seed, double *CV_u, double *CV_x);
void CV_v(double a, double b, double scale, double shift, double num_pow,  double T, double k_u, double k_v, double k_w, double k_x, int step, double dt, double e, double D, int count, int start_time, int distance, int sample_period, int repetetion, int base_seed, double *CV_v, double *CV_x);
void CV_w(double a, double b, double scale, double shift, double num_pow,  double T, double k_u, double k_v, double k_w, double k_x, int step, double dt, double e, double D, int count, int start_time, int distance, int sample_period, int repetetion, int base_seed, double *CV_w, double *CV_x);
double gauss(double uniform1, double uniform2);
void measure_period(int ids_peak[], int size, int sample_period, double dt, double *period, int distance);
void writedata_2d(char *file, char **header, double **data, int rows, int columns);

#endif // FUNCTIONS_H



