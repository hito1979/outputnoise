#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "func.h"  // Include the header for function declarations

//**********************************************************************************************************************************************************************************

int main() {
    //**********************************************************************************
    int N = 3; //the number of harmonics
    int iteration = 1e+7;
    int flag_continue = 0; //flag to continue
    int num_continue = 1000; //the current file

    char base_path[200] = ".."; //the path of folder where it is
    int length = 1000;  //how many sample in one file
    int num_weight = 1e+3; //conditional probability distribution by dividing it into "num_weight" sections
    double T_temp = 1.0; //temprerature for gibbs sampling
    double W = 2.0*M_PI; //omega
    double T = 1.0; //the period
    double epsilon = 1e-1;
    double D = 3.0;
    double k_x = 10.0;
    double tcp = 0.0;
    int seed_uniform = 0;
    char **header;
    char file_sample[200];

    header = (char **)malloc((2 * N + 1) * sizeof(char *));
    for (int i = 0; i < 2 * N + 1; i++) {
        header[i] = (char *)malloc(20 * sizeof(char));
    }

    double **sample = (double**)malloc(sizeof(double*) * (length));
    for (int i = 0; i < length; ++i) {
        sample[i] = (double*)malloc(sizeof(double) * (2 * N + 1));
    }

    for (int i = 0; i < 2*N+1; i++) {

        if (i==0){
            strcpy(header[i], "fitness");
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

    Bounds bounds_x = {-1.0, 1.0}; //boundary for A_n and B_n

    //##################################################################################
    //main

    gibbs_sampling(flag_continue, num_continue, length, header, sample, T_temp, num_weight, iteration, bounds_x, N, tcp, k_x, W, T, epsilon, D, seed_uniform);

    return 0;
}
