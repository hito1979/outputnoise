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
    int repetition = 100; //how many times

    char base_path[200] = ".."; //the path of folder where it is
    double T = 39.7;
    double epsilon = 0.1; //epsilon
    double D = 1e-6;
    double a; //alpha
    double b = 1.0; //beta
    double scale = 1.0; //amplitude of g(w)
    double shift = 0.0; //base level of g(w)
    double num_pow = 1.0; //the degree of polynomial
    double k_u = 0.1;
    double k_v = 0.1;
    double k_w = 0.1;
    double k_x = 1.0;
    double dt = 1e-3; //the size of step for euler method
    int distance; //the minimum distance bwteen peaks
    int start_time; //the time to start detecting peaks
    int num_peak = 1500; //number of peaks
    int sample_period = 1000; //number of periods
    int step; //how many steps for euler method
    double CV_w; //CV of w
    double CV_x; // CV of x
    char file_sample[200];

    distance = (int)(0.8/dt);
    start_time = (int)(100/dt);
    step = (int)(1600/dt);

    // Allocate memory for the header
    char *header[] = {"a", "CV_w", "CV_x"};

    double list_CV[num_sample][3];

    unsigned int previous_seed = 0;
    unsigned int base_seed = 0;

    //##################################################################################
    //main

    double delta = (10.0 - 1.0)/(num_sample - 1);

    for (int i = 0; i < num_sample; i++) {

        base_seed = 100*i;

        CV_w = 0.0;
        CV_x = 0.0;

        a = 1.0 + i * delta;

        CV(a, b, scale, shift, num_pow, T, k_u, k_v, k_w, k_x, step, dt, epsilon, D, num_peak, start_time, distance, sample_period, repetition, base_seed, &CV_w, &CV_x);

        printf("index=%d, CV_w=%.6lf, CV_x=%.6lf\n", i, CV_w, CV_x);

        list_CV[i][0] = a;
        list_CV[i][1] = CV_w;
        list_CV[i][2] = CV_x;
    }

    //save data
    sprintf(file_sample, "%s/Result/CV.csv", base_path); //set your path
    FILE *fp = fopen(file_sample, "w");
    int num_row = num_sample;
    int num_column = 3;
    if (fp == NULL) {
        perror("can't open a csv file");
    }else{
        for (int i = 0; i < num_column; i++) {
            fprintf(fp, "%s", header[i]);
            if (i < num_column - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
        for (int i = 0; i < num_row; i++) {
            for (int j = 0; j < num_column; j++) {
                fprintf(fp, "%.15lf", list_CV[i][j]);
                if (j < num_column - 1) {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("complete writing a csv file: %s\n", file_sample);
    }

    return 0;
}
