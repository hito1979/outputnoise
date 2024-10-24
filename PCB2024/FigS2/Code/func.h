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
typedef struct {
    double lower_bound;
    double upper_bound;
} Bounds;

typedef struct {
    char *strategy;
    int population_size;
    double mutation_factor;
    double crossover_rate;
    int max_generations;
} DEParams;
typedef double (*ObjectiveFunction)(double, double, double[], double, int, double, double, double, double, int, int, int, int, int, int, unsigned int);
typedef void (*Normalise)(double[], int);

void differential_evolution(ObjectiveFunction objective, Normalise normalise, Bounds bounds_x, int dim, DEParams params, int iteration, char *stra, int popsize, double mutation, double crossover, double threshold_convergence, int count_convergence, double a, double b, double k, int step, double W, double dt, double e, double D, int N, int num_sample, int start_time, int distance, int sample_period, int repetition, int length, int seed);
void normalise_vector(double *x, int dim);
double CV(double a, double b, double *params, double k, int step, double W, double dt, double e, double D, int N, int num_sample, int start_time, int distance, int sample_period, int repetition, unsigned int base_seed);
double limitcycle_output(double a, double b, double *A, double *B, double k, double theta, double W, int N);
double gauss(double uniform1, double uniform2);
double f(double theta, double *A, double *B, int N);
void measure_period(int *ids_peak, int size, int sample_period, double dt, double *period, int distance);

// Add other function declarations...

#endif // FUNCTIONS_H



