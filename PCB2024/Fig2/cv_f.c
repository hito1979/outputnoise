#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "SFMT.h" //Mersenne Twister
#include "func.h"  // Include the header for function declarations

//**********************************************************************************************************************************************************************************

int main() {
    //**********************************************************************************
    int N = 5; //the number of harmonics
    int num_sample = 1000; //how many samples 
    int repetition = 100; //how many times

    char base_path[200] = ".."; //the path of folder where it is
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
    double lower_bound = -1.0;
    double upper_bound = 1.0;
    int step; //how many steps for euler method
    double base_seed;
    int seed = 0;
    double CV_x; //CV of x
    char **header;
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


    //header
    header = (char **)malloc((2 * N + 1) * sizeof(char *));
    for (int i = 0; i < 2 * N + 1; i++) {
        header[i] = (char *)malloc(20 * sizeof(char));
    }
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

    double **list_CV = (double**)malloc(sizeof(double*) * (num_sample));
    for (int i = 0; i < num_sample; ++i) {
        list_CV[i] = (double*)malloc(sizeof(double) * (2*N+1));
    }

    //##################################################################################
    //main
    //Mersenne Twister
    sfmt_t sfmt;
    uint64_t generator;
    double uniform_random;
    sfmt_init_gen_rand(&sfmt, seed);

    double A[N], B[N];

    for (int i = 0; i < num_sample; i++) {
        CV_x = 0.0;

        for (int j = 0; j < 2*N; j++) {
            generator = sfmt_genrand_uint64(&sfmt);
            uniform_random = sfmt_to_res53(generator); //first uniform
            X[j] = -1 + uniform_random * 2;
        }
        normalise_vector(X, 2*N);
        for (int j = 0; j < N; j++) {
            A[j] = X[j];
            B[j] = X[N + j];
        }

        base_seed = 100*i;

        CV(a, b, A, B, k_x, step, W, dt, epsilon, D, N, num_peak, start_time, distance, sample_period, repetition, base_seed, &CV_x);

        printf("index=%d, CV_x=%.6lf\n", i, CV_x);

        list_CV[i][0] = CV_x;
        for (int j = 0; j < 2*N; j++) {
            list_CV[i][j+1] = X[j];
        }
    }

    //save data
    sprintf(file_sample, "%s/Result/CV_1.csv", base_path);
    writedata_2d(file_sample, header, list_CV, num_sample, 2*N+1);

    return 0;
}
