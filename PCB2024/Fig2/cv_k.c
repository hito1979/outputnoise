#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "func.h"  // Include the header for function declarations

//**********************************************************************************************************************************************************************************

int main() {
    //**********************************************************************************
    //global variable
    int N = 5;
    int num_sample = 250;
    int repetition = 100;

    //**********************************************************************************
    //definition
    char base_path[200] = "..";
    double W = 2.0*M_PI;
    double epsilon = 1e-1;
    double D = 3.0;
    double a = 1.0; //alpha
    double b = 0.4;  //beta
    double k_x;
    double dt = 1e-3;
    int distance;
    int start_time;
    int num_peak = 1200;
    int sample_period = 1000;
    double lower_bound = -1.0;
    double upper_bound = 1.0;
    int step;
    double pow_k;
    double base_seed;
    double CV_x;
    char file_sample[200];
    double X[2*N];

    distance = (int)(0.8/dt);
    start_time = (int)(100/dt);
    step = (int)(1400/dt);

    //header
    char *header[] = {"k", "CV_x"};

    double **list_CV = (double**)malloc(sizeof(double*) * (num_sample));
    for (int i = 0; i < num_sample; ++i) {
        list_CV[i] = (double*)malloc(sizeof(double) * (2));
    }

    char file[200];
    double normalise_paramter[2*N];
    sprintf(file, "%s/fig2b_blue_round.csv", base_path);
    readdata(file, X);

    double A[N], B[N];

    //A and B
    for (int i = 0; i < N; i++) {
        A[i] = X[i];
        B[i] = X[N + i];
    }

    //##################################################################################
    //main

    double delta = (2.0 - 0.0)/(num_sample - 1);

    for (int i = 0; i < num_sample; i++) {
        CV_x = 0.0;

        pow_k = 0.0 + i * delta;
        k_x = pow(10.0, pow_k);

        base_seed = 100*i;

        CV(a, b, A, B, k_x, step, W, dt, epsilon, D, N, num_peak, start_time, distance, sample_period, repetition, base_seed, &CV_x);

        printf("index=%d, CV_x=%.6lf\n", i, CV_x);

        list_CV[i][0] = pow_k;
        list_CV[i][1] = CV_x;
    }

    //save data
    sprintf(file_sample, "%s/Result/fig2b_blue.csv", base_path);
    writedata_2d(file_sample, header, list_CV, num_sample, 2);

    return 0;
}

