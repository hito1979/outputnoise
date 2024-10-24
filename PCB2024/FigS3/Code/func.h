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

/*
typedef struct {
    double *vector;
    double fitness;
} Individual;
*/

typedef struct {
    char *strategy;
    int population_size;
    double mutation_factor;
    double crossover_rate;
    int max_generations;
} DEParams;
typedef double (*ObjectiveFunction)(double[], int, double, double, double, double, double, double, int, double[], int, double[], int, double[]);
typedef void (*Normalise)(double[], int);

double keep_in_bounds_k(double x, Bounds bounds);
void keep_in_bounds_x(double *x, int dim, Bounds bounds);
void differential_evolution(ObjectiveFunction objective, Normalise normalise, Bounds bounds_x, int dim, DEParams params, int iteration, char *stra, int popsize, double mutation, double crossover, double threshold_convergence, int count_convergence, int N, double W, double T, double e, double D, double a, double b, int num_tcp, double *list_tcp, int num_k, double *list_k, int num_rate, double *list_rate, int length, int seed);
void normalise_vector(double *x, int dim);
double Func(double *params, int N, double W, double T, double e, double D, double a, double b, int num_tcp, double *list_tcp, int num_k, double *list_k, int num_rate, double *list_rate);
//double Func(double *params, int N, double W, double T, double e, double D, double a, double b, int num_tcp, double *list_tcp, int num_k, double *list_k, int num_rate, double *list_rate)
double func1(double tcp, double k, double *A, double *B, int N, double W);
double func2(double tcp, double k, double *A, double *B, int N, double W);
double func3(double tcp, double k, double *A, double *B, int N, double W);
double sgn(double value);
void findMinMax(const double waveform[], int size, int *min_value, int *max_value);
int find_tcp(double *list_output, int num_tcp, double *list_tcp, double h);
double output(double t, double k, double *A, double *B, int N, double W, double a, double b);

// Add other function declarations...

#endif // FUNCTIONS_H



