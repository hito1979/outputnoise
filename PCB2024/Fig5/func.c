// functions.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "SFMT-neon.h" //Mersenne Twister
#include "func.h"  // Include the header for function declarations

// functions.h
#define FUNCTIONS_H
#define MAX_LINE_SIZE 1024

//**********************************************************************************************************************************************************************************
//Gibbs sampling

void gibbs_sampling(int flag_continue, int num_continue, int length, char **header, double **sample, double T_temp, int num_weight, int iteration, Bounds bounds_x, int N, double tcp, double k_x, double W, double T, double epsilon, double D, int seed_uniform) {
    double weights[num_weight], new_CV[num_weight];
    double range = bounds_x.upper_bound - bounds_x.lower_bound;
    int num_file;
    int sample_index, index, seed;
    int num;
    int dim = 2*N;
    double delta = range / (num_weight - 1);
    char base_path[200] = "..";
    double A[N], B[N];
    double CV_x;

    //Mersenne Twister
    sfmt_t sfmt;
    uint64_t generator;
    double uniform_random;

    //seed
    sfmt_init_gen_rand(&sfmt, seed_uniform);

    //initial value
    if (flag_continue == 0){
        for (int i = 0; i < dim; i++) {

            //generate gauss by Mersenne Twister
            generator = sfmt_genrand_uint64(&sfmt);
            uniform_random = sfmt_to_res53(generator); //first uniform

            sample[0][i + 1] = bounds_x.lower_bound + uniform_random * (bounds_x.upper_bound - bounds_x.lower_bound);
        }

        for (int i = 0; i < N; i++) {
            A[i] = sample[0][i + 1];
            B[i] = sample[0][i + N + 1];
        }

        sample[0][0] = CV(&sample[0][1], N, tcp, k_x, W, T, epsilon, D);
        sample_index = 1; //the current parameter
        num = 1; //the iteration
        num_file = 1; //the next file
        printf("iteration=%d\n", 1);
    } else {
        //to restart from the last time
        char file_sample[200];
        sprintf(file_sample, "%s/Result/Sample/sample_%d.csv", base_path, num_continue); //set your path
        readdata_2d(file_sample, sample, length, 2*N+1);
        sample_index = 0;
        num = num_continue*length+1; //the number of sample gotten by the last time
        num_file = num_continue+1; //the file number will be producted next
        int start_dim = (num_continue*length)%dim;

        //to skip some random value to reproduct the last time continue
        int num_skip = dim + (num_continue*length - 1);
        for (int i = 0; i < num_skip; i++) {
            generator = sfmt_genrand_uint64(&sfmt);
            uniform_random = sfmt_to_res53(generator); //first uniform
        }

        for (int j = start_dim; j < dim; j++) {
            double start = clock();
            //to copy the current paramter to the next
            for (int k = 1; k <= dim; k++) {
                sample[sample_index][k] = sample[length - 1][k];
            }

            //generate gauss by Mersenne Twister
            generator = sfmt_genrand_uint64(&sfmt);
            uniform_random = sfmt_to_res53(generator); //first uniform
            calculate_weights(num_weight, weights, new_CV, delta, T_temp, &sample[sample_index][1], j, bounds_x, N, tcp, k_x, W, T, epsilon, D);
            index = get_weighted_random_sample(num_weight, weights, delta, j, uniform_random, bounds_x);
            sample[sample_index][j+1] = bounds_x.lower_bound + index * delta;
            sample[sample_index][0] = new_CV[index];  //update CV

            // Save data every 'length' iterations
            if ((sample_index % (length-1) == 0) && (sample_index != 0)) {
                char file_sample[200];
                sprintf(file_sample, "%s/Result/Sample/sample_%d.csv", base_path, num_file); //set your path
                writedata_2d(file_sample, header, sample, length, 2 * N + 1);
                double end = clock();
                double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
                printf("File %d generated in %lf seconds\n", num_file, cpu_time_used);
                num_file++; //to the next file
                sample_index = 0;
            }else{
                sample_index++;
            }
            num++;
            printf("iteration=%d\n", num);
        }

    }
    double start = clock(); // Start time for file generation
    while (num <= iteration) {
        for (int j = 0; j < dim; j++) {
            if (sample_index == 0){
                double start = clock();
                for (int k = 1; k <= dim; k++) {
                    sample[sample_index][k] = sample[length - 1][k];
                }
            }else{
                for (int k = 1; k <= dim; k++) {
                    sample[sample_index][k] = sample[sample_index-1][k];
                }
            }

            //generate gauss by Mersenne Twister
            generator = sfmt_genrand_uint64(&sfmt);
            uniform_random = sfmt_to_res53(generator); //first uniform
            calculate_weights(num_weight, weights, new_CV, delta, T_temp, &sample[sample_index][1], j, bounds_x, N, tcp, k_x, W, T, epsilon, D);
            index = get_weighted_random_sample(num_weight, weights, delta, j, uniform_random, bounds_x);
            sample[sample_index][j+1] = bounds_x.lower_bound + (double)(index) * delta;
            sample[sample_index][0] = new_CV[index];  //update CV

            // Save data every 'length' iterations
            if ((sample_index % (length-1) == 0) && (sample_index != 0)) {
                char file_sample[200];
                num++;
                printf("iteration=%d\n", num);
                sprintf(file_sample, "%s/Result/Sample/sample_%d.csv", base_path, num_file); //set your path
                writedata_2d(file_sample, header, sample, length, 2 * N + 1);
                double end = clock();
                double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
                printf("File %d generated in %lf seconds\n", num_file, cpu_time_used);
                num_file++;
                sample_index = 0;
            }else{
                num++;
                sample_index++;
                printf("iteration=%d\n", num);
            }
            if (num >= iteration) {
                break;
            }
        }

        if (num >= iteration) {
            break;
        }
    }

    if (sample_index > 0 && sample_index % (length-1) != 0) {
        char file_sample[200];
        sprintf(file_sample, "%s/Result/sample_%d.csv", base_path, num_file); //set your path
        writedata_2d(file_sample, header, sample, sample_index, 2 * N + 1);
    }
}

void calculate_weights(int num_weight, double* weights, double *new_CV, double delta, double T_temp, double* current_params, int param_index, Bounds bounds_x, int N, double tcp, double k_x, double W, double T, double e, double D) {
    double range = bounds_x.upper_bound - bounds_x.lower_bound;
    double cv_value;
    int current_index = param_index;
    double local_current_params[2*N];

    for (int i = 0; i < 2*N; i++) {
        local_current_params[i] = current_params[i];
    }

    #pragma omp parallel for private(cv_value) firstprivate(local_current_params)
    for (int i = 0; i < num_weight; i++) {
        local_current_params[current_index] = bounds_x.lower_bound + i * delta;
        cv_value = CV(local_current_params, N, tcp, k_x, W, T, e, D);
        weights[i] = exp(-cv_value / T_temp);
        new_CV[i] = cv_value;
    }
}

int get_weighted_random_sample(int num_weight, double* weights, double delta, int param_index, double uniform_random, Bounds bounds_x) {
    double sum_weights = 0.0;
    for (int i = 0; i < num_weight; i++) {
        sum_weights += weights[i];
    }

    double random_value = uniform_random * sum_weights;
    double cumulative_weight = 0.0;
    int sample_index = num_weight - 1;
    for (int i = 0; i < num_weight; i++) {
        cumulative_weight += weights[i];
        if (cumulative_weight >= random_value) {
            sample_index = i;
            break;
        }
    }
    return sample_index;
}

//****************************************************************************************
double CV(double *params, int N, double tcp, double k, double W, double T, double epsilon, double D) {
    //defenition
    double A[N], B[N];

    //A_n and B_n
    for (int i = 0; i < N; i++) {
        A[i] = params[i];
        B[i] = params[N + i];
    }

    double f1 = func1(N, tcp, k, W, A, B);
    double f2 = func2(N, tcp, k, W, A, B);
    double f3 = func3(N, tcp, k, W, A, B);
    double R1 = D * T / pow(W, 2.0);
    double R2 = D * (1 - exp(-k * T)) / pow(f1, 2.0) * f2;
    double R3 = -D * (1 - exp(-k * T)) / (W * f1) * f3;
    double CV = epsilon * sqrt(R1 + R2 + 2 * R3) / T;

    return CV;
}

double func1(int N, double tcp, double k, double W, double *A, double *B){

    double Sum, A_n, B_n, A_m, B_m, Phi_n, Phi_m;
    int M = N;
    
    Sum = 0.0;


    for (int n = 1; n <= N; n++){

        A_n = A[n-1];
        B_n = B[n-1];

        if (!(A_n == 0 && B_n == 0)){

            if ((n*W*A_n+k*B_n) != 0){
                Phi_n = n*W*tcp + atan((k*A_n - n*W*B_n)/(n*W*A_n+k*B_n)) + M_PI/2*(1.0 - sgn(n*W*A_n+k*B_n));
            }else{
                Phi_n = n*W*tcp + sgn(k*A_n - n*W*B_n)*M_PI/2;
            }            

            Sum = Sum + n*W*sqrt((pow(A_n, 2) + pow(B_n, 2))/(pow(k, 2) + pow(n*W, 2)))*cos(Phi_n);

        }

    }

    return Sum;

}

double func2(int N, double tcp, double k, double W, double *A, double *B){

    double Sum, sum1, sum2, sum3, A_n, B_n, A_m, B_m, Phi_n, Phi_m;
    int M = N;

    Sum = 0.0;

    for (int n = 1; n <= N; n++){

        A_n = A[n-1];
        B_n = B[n-1];

        if (!(A_n == 0 && B_n == 0)){

            if ((n*W*A_n+k*B_n) != 0){
                Phi_n = n*W*tcp + atan((k*A_n - n*W*B_n)/(n*W*A_n+k*B_n)) + M_PI/2*(1 - sgn(n*W*A_n+k*B_n));
            }else{
                Phi_n = n*W*tcp + sgn(k*A_n - n*W*B_n)*M_PI/2;
            }

            for (int m = 1; m <= M; m++){

                A_m = A[m-1];
                B_m = B[m-1];

                if (!(A_m == 0 && B_m == 0)){

                    if ((m*W*A_m+k*B_m) != 0){
                        Phi_m = m*W*tcp + atan((k*A_m - m*W*B_m)/(m*W*A_m+k*B_m)) + M_PI/2*(1 - sgn(m*W*A_m+k*B_m));
                    }else{
                        Phi_m = m*W*tcp + sgn(k*A_m - m*W*B_m)*M_PI/2;
                    }

                    sum1 = n*m*sqrt(((pow(A_n, 2) + pow(B_n, 2))*(pow(A_m, 2) + pow(B_m, 2)))/((pow(k, 2) + pow(n*W, 2))*(pow(k, 2) + pow(m*W, 2))));
                    sum2 = 1/(4*pow(k, 2) + pow((n - m)*W, 2)) * ((n - m)*W*sin(Phi_n - Phi_m) + 2*k*cos(Phi_n - Phi_m));
                    sum3 = 1/(4*pow(k, 2) + pow((n + m)*W, 2)) * ((n + m)*W*sin(Phi_n + Phi_m) + 2*k*cos(Phi_n + Phi_m));

                    Sum = Sum + sum1*(sum2 + sum3);

                }

            }

        }
    }

    return Sum;

}

double func3(int N, double tcp, double k, double W, double *A, double *B){

    double Sum, sum1, sum2, A_n, B_n, Phi_n;
    int M = N;

    Sum = 0.0;

    for (int n = 1; n <= N; n++){

        A_n = A[n-1];
        B_n = B[n-1];

        if (!(A_n == 0 && B_n == 0)){

            if ((n*W*A_n+k*B_n) != 0){
                Phi_n = n*W*tcp + atan((k*A_n - n*W*B_n)/(n*W*A_n+k*B_n)) + M_PI/2*(1 - sgn(n*W*A_n+k*B_n));
            }else{
                Phi_n = n*W*tcp + sgn(k*A_n - n*W*B_n)*M_PI/2;
            }

            sum1 = n*sqrt((pow(A_n,2)+pow(B_n,2))/(pow(k,2)+pow(n*W,2)))*1/(pow(k,2)+pow(n*W,2));
            sum2 = n*W*sin(Phi_n)+k*cos(Phi_n);

            Sum = Sum + sum1*sum2;

        }

    }

    return Sum;

}

double sgn(double value) {
    if (value >= 0) {
        return 1;
    } else {
        return -1;
    }
}


double output(double t, int N, double tcp, double a, double b, double k, double W, double *A, double *B) {
    double Sum = a / k;

    #pragma omp parallel for reduction(+:Sum)
    for (int n = 1; n <= N; ++n) {
        Sum += b * 1 / (pow(k, 2) + pow(n*W, 2)) * ((n*W*A[n-1] + k*B[n-1]) * sin(n*W*t) + (k*A[n-1] - n*W*B[n-1]) * cos(n*W*t));
    }

    return Sum;
}


//****************************************************************************************

void writedata_2d(char *file, char **header, double **data, int rows, int columns) {
    FILE *fp = fopen(file, "w");

    if (fp == NULL) {
        perror("fail to open a file");
        return;
    }

    for (int i = 0; i < columns; i++) {
        fprintf(fp, "%s", header[i]);
        if (i < columns - 1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "\n");

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            fprintf(fp, "%.15lf", data[i][j]);
            if (j < columns - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);

    printf("Write data on CSV file: %s\n", file);
}

void readdata_2d(const char *file, double **dataArray, int rows, int columns) {
    FILE *fp = fopen(file, "r");

    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    char line[MAX_LINE_SIZE];
    int row = 0;

    // Skip the header (assuming it's the first line)
    if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return; // Error or empty file
    }

    while (fgets(line, sizeof(line), fp) != NULL && row < rows) {
        char *token = strtok(line, ",");
        int col = 0;
        while (token != NULL && col < columns) {
            dataArray[row][col] = atof(token);
            token = strtok(NULL, ",");
            col++;
        }
        row++;
    }

    fclose(fp);
}

