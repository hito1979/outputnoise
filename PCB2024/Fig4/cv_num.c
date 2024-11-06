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
    int N = 10; //the number of harmonics
    int num_sample = 21; //how many samples 
    int repetition = 100; //how many times

    char base_path[200] = ".."; //the path of folder where it is
    double W = 2.0*M_PI; //omega
    double epsilon = 1e-1;
    double D = 0.1;
    double a = 4.0; //alpha
    double b = 1.0; //beta
    double k_x;
    double dt = 1e-3; //the size of step for euler method
    int distance; //the minimum distance bwteen peaks
    int start_time; //the time to start detecting peaks
    int num_peak = 1200; //number of peaks
    int sample_period = 1000; //number of periods
    double rate = 0.5; //rate of oscillation range
    int step; //how many steps for euler method
    int seed = 0;
    double pow_k;
    double CV_x, cv_x;
    int i;
    char file_sample[200];
    double X[2*N];

    distance = (int)(0.8/dt);
    start_time = (int)(100/dt);
    step = (int)(1400/dt);

    double list_pow_k[] = {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0
    };

    char *list_folder_k[] = {
        "0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7",
        "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5",
        "1.6", "1.7", "1.8", "1.9", "2.0"
    };

    //header
    char *header[] = {"CV_x"};

    double list_CV[repetition];

    char file[200];
    sprintf(file, "parameter_blue.csv"); //set your path
    readdata(file, X);

    //##################################################################################
    //main

    for (int i = 0; i < num_sample; i++) {
        
        pow_k = list_pow_k[i];
        k_x = pow(10.0, pow_k);

        for (int j = 0; j < repetition; j++) {
            seed = seed + 1;
            CV_x = CV(a, b, X, N, k_x, step, W, dt, epsilon, D, start_time, distance, num_peak, sample_period, rate, seed);
            list_CV[j] = CV_x;
        }

        //save data
        sprintf(file_sample, "%s/Result/k_%s/CV.csv", base_path, list_folder_k[i]);
        writedata_column(file_sample, header[0], list_CV, repetition);
        
    }

    return 0;
}
