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
void euler_maruyam(double a, double b, double *A, double *B, double k, int step, double W, double dt, double e, double D, int N, double *arr_theta, double *arr_x, int seed);
double limitcycle_output(double a, double b, double *A, double *B, double k, double theta, double W, int N);
double gauss(double uniform1, double uniform2);
double f(double theta, double *A, double *B, int N);
double output(double t, double a, double b, double *A, double *B, double k, double W, double N);
double CV(double a, double b, double *X, int N, double k_x, int step, double W, double dt, double e, double D, int start_time, int distance, int num_peak, int sample_period, double rate, int seed);
void measure_period(int *ids_checkpoint, int num_sample, double dt, double *period, int distance);
void detect_peaktrough(double *X, int *ids_peak, int *ids_trough, int count, int start_time, int distance, double dt);
int find_max_index(double *X, int start, int end);
int find_min_index(double *X, int start, int end);
void find_checkpoint(double *X, int *ids_peak, int *ids_trough, int size, double threshold, int *ids_checkpoint);
double threshold(double rate, double a, double b, double *A, double *B, double k, double W, int N);
double find_tcp(double h, double a, double b, double *A, double *B, double k, double W, int N);
void findMinMax(const double waveform[], int size, int *min_value, int *max_value);
void normalise_vector(double *x, int dim);
double CV_ana(double* params, double T, double k, double W, double e, double D, int N, double tcp);
double func1(double tcp, double k, double W, double *A, double *B, int N);
double func2(double tcp, double k, double W, double *A, double *B, int N);
double func3(double tcp, double k, double W, double *A, double *B, int N);
double sgn(double value);
void writedata_column(char *file, char *header, double *data, int length);
void writedata_row(char *file, char **header, double *data, int length);
void writedata_2d(char *file, char **header, double **data, int rows, int columns);
void readdata(const char *file, double *dataArray);

#endif // FUNCTIONS_H



