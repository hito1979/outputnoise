// functions.h
#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#define MAX_LINE_SIZE 1024

// Global variable declarations

// Function declarations
typedef struct {
    double lower_bound;
    double upper_bound;
} Bounds;
void gibbs_sampling(int flag_continue, int num_continue, int length, char **header, double **sample, double T_temp, int num_weight, int iteration, Bounds bounds_x, int N, double tcp, double k_x, double W, double T, double epsilon, double D, int seed_uniform);
void calculate_weights(int num_weight, double* weights, double *new_CV, double delta, double T_temp, double* current_params, int param_index, Bounds bounds_x, int N, double tcp, double k_x, double W, double T, double e, double D);
int get_weighted_random_sample(int num_weight, double* weights, double delta, int param_index, double uniform_random, Bounds bounds_x);
double CV(double *params, int N, double tcp, double k, double W, double T, double epsilon, double D);
double func1(int N, double tcp, double k, double W, double *A, double *B);
double func2(int N, double tcp, double k, double W, double *A, double *B);
double func3(int N, double tcp, double k, double W, double *A, double *B);
double sgn(double value);
double output(double t, int N, double tcp, double a, double b, double k, double W, double *A, double *B);
void writedata_2d(char *file, char **header, double **data, int rows, int columns);
void readdata_2d(const char *file, double **dataArray, int rows, int columns);

// Add other function declarations...

#endif // FUNCTIONS_H



