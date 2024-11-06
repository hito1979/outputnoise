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
    int N = 5; //the number of harmonics
    int num_sample = 250; //how many samples 
    int repetition = 100; //how many times

    char base_path[200] = ".."; //the path of folder where it is
    double W = 2.0*M_PI; //omega
    double epsilon = 1e-1;
    double D = 3.0;
    double a; //alpha
    double b = 0.4; //beta
    double k_x = 10.0;
    double dt = 1e-3; //the size of step for euler method
    int distance; //the minimum distance bwteen peaks
    int start_time; //the time to start detecting peaks
    int num_peak = 1200; //number of peaks
    int sample_period = 1000; //number of periods
    int step; //how many steps for euler method
    double base_seed;
    double CV_x; // CV of x
    char file_sample[200];
    double X[2*N];

    distance = (int)(0.8/dt);
    start_time = (int)(100/dt);
    step = (int)(1400/dt);

    //header
    char *header[] = {"a", "CV_x"};

    double **list_CV = (double**)malloc(sizeof(double*) * (num_sample));
    for (int i = 0; i < num_sample; ++i) {
        list_CV[i] = (double*)malloc(sizeof(double) * (2));
    }

    char file[200];
    double normalise_paramter[2*N];
    sprintf(file, "parameter_blue.csv"); //set your path for CSV file of red or blue waveform
    readdata(file, X);

    double A[N], B[N];

    //A and B
    for (int i = 0; i < N; i++) {
        A[i] = X[i];
        B[i] = X[N + i];
    }

    //##################################################################################
    //main

    double delta = (10.0 - 1.0)/(num_sample - 1);

    for (int i = 0; i < num_sample; i++) {
        CV_x = 0.0;

        a = 1.0 + i * delta;

        base_seed = 100*i;

        CV(a, b, A, B, k_x, step, W, dt, epsilon, D, N, num_peak, start_time, distance, sample_period, repetition, base_seed, &CV_x);

        printf("index=%d, CV_x=%.6lf\n", i, CV_x);

        list_CV[i][0] = a;
        list_CV[i][1] = CV_x;
    }

    //save data
    sprintf(file_sample, "%s/Result/fig2b_blue.csv", base_path); //set your path
    writedata_2d(file_sample, header, list_CV, num_sample, 2);

    return 0;
}
