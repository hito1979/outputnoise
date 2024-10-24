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
    int N = 3; //the number of harmonics
    int iteration = 1e+6; 
    int flag_continue = 0; //flag to continue from the current file
    int num_continue = 234; //num of the current file
    int repetition = 100; //how many repetetion

    int length = 10; //how many sample in one file
    int num_weight = 50; //conditional probability distribution by dividing it into "num_weight" sections
    double T_temp = 1.0; //temprerature for gibbs sampling
    double W = 2.0*M_PI; //omega
    double epsilon = 1e-1;
    double D = 3.0;
    double a = 3.0; //alpha
    double b = 1.0; //beta
    double k_x = 10.0;
    double dt = 1e-3; //the size of step for euler method
    int distance; //the minimum distance bwteen peaks
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

    header = (char **)malloc((2 * N + 1) * sizeof(char *));
    for (int i = 0; i < 2 * N + 1; i++) {
        header[i] = (char *)malloc(20 * sizeof(char));
    }

    double **sample = (double**)malloc(sizeof(double*) * (length));
    for (int i = 0; i < length; ++i) {
        sample[i] = (double*)malloc(sizeof(double) * (2 * N + 1));
    }


    //header
    for (int i = 0; i < 2*N+1; i++) {

        if (i==0){
            strcpy(header[i], "CV");
        }

        else if (0 < i && i <= N){
            char temp[20];
            sprintf(temp, "%s%d", "A_", i);
            strcpy(header[i], temp);
        }

        else {
            char temp[20];
            sprintf(temp, "%s%d", "B_", i-N);
            strcpy(header[i], temp);
        }

    }

    Bounds bounds_x = {-1.0, 1.0};

    //##################################################################################
    //main
    gibbs_sampling(flag_continue, num_continue, length, header, sample, T_temp, num_weight, iteration, bounds_x, a, b, N, k_x, step, W, dt, epsilon, D, start_time, distance, num_peak, sample_period, repetition, seed_uniform, &base_seed);
    
    return 0;
}
