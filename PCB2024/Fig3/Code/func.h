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
typedef struct {
    double lower_bound;
    double upper_bound;
} Bounds;

void gibbs_sampling(int flag_continue, int num_continue, int length, char **header, double **sample, double T_temp, int num_weight, int iteration, Bounds bounds_x, double a, double b, int N, double k_x, int step, double W, double dt, double e, double D, int start_time, double distance, int num_peak, int sample_period, int repetition, int seed_uniform, unsigned int *base_seed);
void calculate_weights(int num_weight, double* weights, double *new_CV, double delta, double T_temp, double* current_params, int param_index, Bounds bounds_x, double a, double b, int N, double k_x, int step, double W, double dt, double e, double D, int start_time, double distance, int num_peak, int sample_period, int repetition, unsigned int *base_seed);
int get_weighted_random_sample(int num_weight, double* weights, double delta, int param_index, double uniform_random, Bounds bounds_x);
void CV(double a, double b, double *A, double *B, double k, int step, double W, double dt, double e, double D, int N, int count, int start_time, int distance, int sample_period, int repetition, unsigned int base_seed, double *CV_x);
double limitcycle_output(double a, double b, double *A, double *B, double k, double theta, double W, int N);
double gauss(double uniform1, double uniform2);
double f(double theta, double *A, double *B, int N);
void measure_period(int ids_peak[], int size, int sample_period, double dt, double *period, int distance);
void normalize_samples(double **data, int rows, int columns);
void writedata_2d(char *file, char **header, double **data, int rows, int columns);
void readdata_2d(const char *file, double **dataArray, int rows, int columns);

// Add other function declarations...

#endif // FUNCTIONS_H



