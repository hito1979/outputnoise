#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "func.h"  // Include the header for function declarations

//**********************************************************************************************************************************************************************************

int main() {
    //**********************************************************************************
    int num = 10; //n-th trial
    int iteration = 1e+2;
    int N = 5; //the number of harmonics
    int repetition = 100;

    char base_path[200] = ".."; //the path where this code is
    double W = 2.0*M_PI; //omega
    double T = 1.0; //the period
    double e = 0.1; //epsilon
    double D = 3.0;
    double a = 3.0; //alpha
    double b = 1.0; //beta
    double k_x = 10.0;
    char stra[100] = "rand1bin"; //the strategy
    int popsize = 50; //pouplation size
    double mutation = 0.9;
    double crossover = 0.8;
    double threshold_convergence = 0; //Threshold for the rate of change between generations for convergence determination
    int count_convergence; //
    int length = 10; //the number of samples in one file
    int seed;
    count_convergence = (int)(0.2*iteration); //maximum of convergence

    double dt = 1e-3;
    int distance; //minimum period between peaks
    int start_time; //the time to start detecting peaks
    int num_peak = 1200; //number of peaks
    int sample_period = 1000; //number of periods
    int step; //how many steps for euler method
    unsigned int base_seed = 0;
    int seed_uniform = 0;
    double CV_x, cv_x;
    char **header;
    char file_sample[200];

    
    distance = (int)(0.8/dt);
    start_time = (int)(100/dt);
    step = (int)(1400/dt);

    //hyperparameter
    Bounds bounds_x = {-1.0, 1.0}; //boundary of A_n and B_n
    DEParams params = {stra, popsize, mutation, crossover, iteration};  //{戦略名, 個体数, 突然変異, 交叉率, 最大世代数}

    //##################################################################################
    //mian
    printf("DE\n");
    printf("%d\n", num);
    printf("N=%d\n", N);
    printf("Iteration= %d\n", iteration);

    //defferential evolution
    seed = num;
    differential_evolution(CV, normalise_vector, bounds_x, 2*N, params, iteration, stra, popsize, mutation, crossover, threshold_convergence, count_convergence, a, b, k_x, step, W, dt, e, D, N, num_peak, start_time, distance, sample_period, repetition, length, seed);

    return 0;
}
