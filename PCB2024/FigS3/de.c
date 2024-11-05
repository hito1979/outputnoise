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
    int iteration = 2e+5;
    int N = 10; //the number of harmonics

    char base_path[200] = ".."; //the path where this code is
    double W = 2.0*M_PI; //omega
    double T = 1.0; //the period
    double e = 0.1; //epsilon
    double D = 3.0; 
    double a = 4.0; //alpha
    double b = 1.0; //beta
    int num_tcp = 1e+3;
    double list_tcp[1000];
    double max_tcp = 1.0;
    double min_tcp = 0.0;
    int num_k = 10;
    double list_k[10];
    double pow_max_k = 2.0;
    double pow_min_k = 0.0;
    double max_rate = 0.8;
    double min_rate = 0.2;
    int num_rate = 10;
    double list_rate[10];
    char stra[100] = "rand1bin";
    int popsize = 50;
    double mutation = 0.9;
    double crossover = 0.8;
    double threshold_convergence = 1e-7; //Threshold for the rate of change between generations for convergence determination
    int count_convergence; //
    int length = 1000;
    int seed;
    count_convergence = (int)(0.2*iteration); //maximum of convergence
    
    //make tcp list
    double step_tcp = 1.0 / (num_tcp - 1);
    for (int i = 0; i < num_tcp; i++){
        list_tcp[i] = i*step_tcp;
    }

    //make k list
    double step_k = (pow_max_k - pow_min_k) / (num_k - 1);
    for (int i = 0; i < num_k; i++) {
        list_k[i] = pow(10, pow_min_k + i * step_k);
    }

    //make rate list
    double step_rate = (max_rate - min_rate) / (num_rate - 1);
    for (int i = 0; i < num_rate; i++){
        list_rate[i] = min_rate + i * step_rate;
    }

    //hyperparameter
    Bounds bounds_x = {-1.0, 1.0};
    DEParams params = {stra, popsize, mutation, crossover, iteration};  //{戦略名, 個体数, 突然変異, 交叉率, 最大世代数}

    //##################################################################################
    //mian
    printf("DE\n");
    printf("%d\n", num);
    printf("N=%d\n", N);
    printf("Iteration= %d\n", iteration);

    //defferential evolution
    seed = num;
    differential_evolution(Func, normalise_vector, bounds_x, 2*N, params, iteration, stra, popsize, mutation, crossover, threshold_convergence, count_convergence, N, W, T, e, D, a, b, num_tcp, list_tcp, num_k, list_k, num_rate, list_rate, length, seed);

    return 0;
}
