#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "func.h"  // Include the header for function declarations


//**********************************************************************************************************************************************************************************

int main() {
    //**********************************************************************************
    int N = 10; //the number of harmonics
    int num_sample = 1e+4; //how many samples 

    char base_path[200] = ".."; //the path of folder where it is
    double T = 1.0; //peirod
    double W = 2.0*M_PI; //omega
    double epsilon = 1e-1;
    double D = 0.1;
    double a = 4.0; //alpha
    double b = 1.0; //beta
    double k_x;
    double rate = 0.5;
    int step; //how many steps for euler method
    double pow_k;
    double tcp;
    double h;

    
    char file_sample[200];
    double X[2*N];
    double list_pow_k[num_sample];
    
    double **list_CV = (double**)malloc(sizeof(double*) * (num_sample));
    for (int i = 0; i < num_sample; ++i) {
        list_CV[i] = (double*)malloc(sizeof(double) * (2));
    }

    // Calculate the step size
    double delta = 2.0 / (num_sample - 1);
    for (int i = 0; i < num_sample; i++) {
        list_pow_k[i] = i * delta;
    }

    //header
    char *header[] = {"k", "CV_x"};

    char file[200];
    sprintf(file, "%s/fig3b_red_round.csv", base_path); //set your path
    readdata(file, X);

    double A[N], B[N];
    for (int i = 0; i < N; i++) {
        A[i] = X[i];
        B[i] = X[N + i];
    }

    //##################################################################################
    //main

    for (int i = 0; i < num_sample; i++) {

        pow_k = list_pow_k[i];
        k_x = pow(10.0, pow_k);

        printf("pow_k=%lf\n", pow_k);

        h = threshold(rate, a, b, A, B, k_x, W, N);
        tcp = find_tcp(h, a, b, A, B, k_x, W, N);
        list_CV[i][0] = k_x;
        list_CV[i][1] = CV_ana(X, T, k_x, W, epsilon, D, N, tcp); 
    }

    sprintf(file_sample, "%s/Result/CV_ana.csv", base_path); //set your path
    writedata_2d(file_sample, header, list_CV, num_sample, 2);

    return 0;
}
