#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include "func.h"  // Include the header for function declarations


//**********************************************************************************************************************************************************************************

int main() {
    //**********************************************************************************
    int num_sample = 250; //how many samples
    int repetition = 100;

    char base_path[200] = ".."; //the path of folder where it is
    double T = 39.7;
    double epsilon = 1e-1; //epsilon
    double D = 1e-6;
    double a = 1.0; //alpha
    double b; //beta
    double scale = 1.0; //amplitude of g
    double shift = 0.0; //base level of g
    double num_pow = 1;
    double k_u = 0.1;
    double k_v = 0.1;
    double k_w = 0.1;
    double k_x = 10.0;
    double dt = 1e-3; //the size of step for euler method
    int distance;
    int start_time; //the time to start detecting peaks
    int num_peak = 1200; //number of peaks
    int sample_period = 1000; //number of periods
    int step; //how many steps for euler method
    double CV_w; //CV of u
    double CV_x; //CV of x
    char file_sample[200];

    distance = (int)(0.8/dt);
    start_time = (int)(100/dt);
    step = (int)(1400/dt);


    // Allocate memory for the header
    char *header[] = {"b", "CV_w", "CV_x"};

    double **list_CV = (double**)malloc(sizeof(double*) * (num_sample));
    for (int i = 0; i < num_sample; ++i) {
        list_CV[i] = (double*)malloc(sizeof(double) * (3));
    }

    unsigned int previous_seed = 0;
    unsigned int base_seed = 0;

    //##################################################################################
    //main

    double delta = (10.0 - 1.0)/(num_sample - 1);

    int seed;

    for (int i = 0; i < num_sample; i++) {

        base_seed = 100*i;

        CV_w = 0.0;
        CV_x = 0.0;

        b = 1.0 + i * delta;

        CV_w(a, b, scale, shift, num_pow, T, k_u, k_v, k_w, k_x, step, dt, epsilon, D, num_peak, start_time, distance, sample_period, repetition, base_seed, &CV_w, &CV_x);

        printf("index=%d, CV_w=%.6lf, CV_x=%.6lf\n", i, CV_w, CV_x);

        list_CV[i][0] = b;
        list_CV[i][1] = CV_w;
        list_CV[i][2] = CV_x;
    }

    //save data
    sprintf(file_sample, "%s/Result/CV.csv", base_path);
    writedata_2d(file_sample, header, list_CV, num_sample, 3);

    return 0;
}
