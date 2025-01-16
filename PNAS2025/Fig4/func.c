#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include "SFMT.h" //Mersenne Twister
#include "func.h"  // Include the header for function declarations
#define MAX_LINE_SIZE 1024

//####################################################################################################################################################################
void euler_maruyam(double a, double b, double *A, double *B, double k, int step, double W, double dt, double e, double D, int N, double *arr_theta, double *arr_x, int seed) {

    //initial value
    arr_theta[0] = 0.0;
    arr_x[0] = limitcycle_output(a, b, A, B, k, arr_theta[0], W, N);

    //combo
    double combo1 = W * dt;
    double combo2 = e * sqrt(D*dt);
    double r;
    double uniform1, uniform2, uniform_random;

    //Mersenne Twister
    sfmt_t sfmt;
    uint64_t generator;

    //seed
    sfmt_init_gen_rand(&sfmt, seed);

    //set the first peak
    for (int i = 0; i < step-1; i++) {

        //generate gauss by Mersenne Twister
        generator = sfmt_genrand_uint64(&sfmt);
        uniform1 = sfmt_to_res53(generator); //first uniform
        generator = sfmt_genrand_uint64(&sfmt);
        uniform2 = sfmt_to_res53(generator); //second uniform
        r = gauss(uniform1, uniform2);

        arr_theta[i + 1] = arr_theta[i] + combo1 + r * combo2;
        arr_x[i + 1] = arr_x[i] + dt * (a + b * f(arr_theta[i], A, B, N) - k * arr_x[i]);
    }
}

double limitcycle_output(double a, double b, double *A, double *B, double k, double theta, double W, int N) {
    double Sum = a / k;
    for (int n = 1; n <= N; n++) {
        double term1 = (n * W * A[n - 1] + k * B[n - 1]) * sin(n * theta);
        double term2 = (k * A[n - 1] - n * W * B[n - 1]) * cos(n * theta);
        Sum += b / (pow(k, 2) + pow(n * W, 2)) * (term1 + term2);
    }
    return Sum;
}

double gauss(double uniform1, double uniform2) {

    double z = sqrt(-2 * log(uniform1)) * cos(2 * M_PI * uniform2);
    return z;
}

double f(double theta, double *A, double *B, int N) {
    double Sum = 0.0;
    for (int n = 1; n <= N; n++) {
        Sum += A[n - 1] * cos(n * theta) + B[n - 1] * sin(n * theta);
    }
    return Sum;
}

double output(double t, double a, double b, double *A, double *B, double k, double W, double N) {
    double Sum = a / k;

    for (int n = 1; n <= N; ++n) {
        Sum += b * 1 / (pow(k, 2) + pow(n*W, 2)) * ((n*W*A[n-1] + k*B[n-1]) * sin(n*W*t) + (k*A[n-1] - n*W*B[n-1]) * cos(n*W*t));
    }

    return Sum;
}

double CV(double a, double b, double *X, int N, double k_x, int step, double W, double dt, double e, double D, int start_time, int distance, int num_peak, int sample_period, double rate, int seed) {
    double *arr_theta = (double *)malloc(step * sizeof(double));
    double *arr_x = (double *)malloc(step * sizeof(double));
    double *period = (double *)malloc((num_peak+100) * sizeof(double));
    int *ids_peak = (int *)malloc((num_peak+100) * sizeof(int));
    int *ids_trough = (int *)malloc((num_peak+100) * sizeof(int));
    int *ids_checkpoint = (int *)malloc((num_peak+100) * sizeof(int));

    double A[N], B[N];
    double mean, std, CV_x, h;

    //A and B
    for (int i = 0; i < N; i++) {
        A[i] = X[i];
        B[i] = X[N + i];
    }

    //euler method
    euler_maruyam(a, b, A, B, k_x, step, W, dt, e, D, N, arr_theta, arr_x, seed);
    detect_peaktrough(arr_x, ids_peak, ids_trough, num_peak, start_time, distance, dt);
    h = threshold(rate, a, b, A, B, k_x, W, N);
    find_checkpoint(arr_x, ids_peak, ids_trough, num_peak, h, ids_checkpoint);
    measure_period(ids_checkpoint, num_peak+100, dt, period, distance);
    mean = 0.0;
    std = 0.0;
    for (int i = 0; i < sample_period; i++) {
        mean += period[i];
    }
    mean /= sample_period;
    for (int i = 0; i < sample_period; i++) {
        std += pow(period[i] - mean, 2);
    }
    std = sqrt(std / sample_period);
    CV_x =  std / mean;

    //Free allocated memory
    free(arr_theta);
    free(arr_x);
    free(period);
    free(ids_peak);
    free(ids_trough);
    free(ids_checkpoint);

    printf("cv_x=%lf\n", CV_x);

    return CV_x;
}

void measure_period(int *ids_checkpoint, int num_sample, double dt, double *period, int distance) {
    double diff;
    // チェックポイント間の期間を計算
    for (int i = 0; i < num_sample; i++) {
        diff = (ids_checkpoint[i + 1] - ids_checkpoint[i])*dt;
        if((distance*dt <= diff) && (diff <= (2.0-distance*dt))){
            period[i] = diff;
        }
    }
}

void detect_peaktrough(double *X, int *ids_peak, int *ids_trough, int count, int start_time, int distance, double dt) {
    int flag = 1;
    int i, j, error;

    int num_index;
    int start, end, new_peak, getout;
    int id_next_peak, id_next_trough;
    int previous_trough, previous_peak;
    int min_period = distance;
    int max_period = (int)(2.0 / dt) - distance;

    //set first trough
    start = start_time; //to avoid zsh segmentation
    end = start_time + (int)(5.0/dt);
    ids_trough[0] = find_min_index(X, start, end);
    
    //set second trough
    previous_trough = ids_trough[0];
    id_next_trough = 0;
    while(id_next_trough == 0){
        start = previous_trough + min_period;
        end = previous_trough + max_period;
        id_next_trough = find_min_index(X, start, end);
        previous_trough = previous_trough + (max_period - min_period);
    }
    ids_trough[1] = id_next_trough;

    //set first peak
    start = ids_trough[0];
    end = ids_trough[1];
    ids_peak[0] = find_max_index(X, start, end);

    //set second peak
    start =  ids_trough[1] > ids_peak[0] + min_period ? ids_trough[1] : ids_peak[0] + min_period;
    end = ids_peak[0] + max_period;
    ids_peak[1] = find_max_index(X, start, end);

    for (i = 1; i <= count; i++){

        //set i th trough
        start =  ids_peak[i] > ids_trough[i] + min_period ? ids_peak[i] : ids_trough[i] + min_period;
        end = ids_trough[i] + max_period;
        id_next_trough = 0;
        while(id_next_trough == 0){
            id_next_trough = find_min_index(X, start, end);
            start = end;
            end = end + (int)(1.0/dt);
        }
        ids_trough[i+1] = id_next_trough;

        //set i th peak
        start =  ids_trough[i+1] > ids_peak[i] + min_period ? ids_trough[i+1] : ids_peak[i] + min_period;
        end = ids_peak[i] + max_period;
        ids_peak[i+1] = find_max_index(X, start, end);

    }
}

int find_max_index(double *X, int start, int end) {
    int max_index = 0;
    double max_value = -INFINITY;
    int error;
    for (int i = start + 1; i <= end; i++) {
        error = 0;
        if (X[i] > max_value) {
            for (int j = 0; j <= 20; j++){
                if (X[i] < X[i-10+j]){
                    error = 1;
                }
            }
            if (error == 0){
                max_value = X[i];
                max_index = i;
            }
        }
    }
    return max_index;
}

int find_min_index(double *X, int start, int end) {
    int min_index = 0;
    double min_value = INFINITY;
    int error;
    for (int i = start + 1; i <= end; i++) {
        error = 0;
        if (X[i] < min_value) {
            for (int j = 0; j <= 20; j++) {
                if (X[i] > X[i-10+j]) {
                    error = 1;
                }
            }
            if (error == 0) {
                min_value = X[i];
                min_index = i;
            }
        }
    }
    return min_index;
}

void find_checkpoint(double *X, int *ids_peak, int *ids_trough, int size, double threshold, int *ids_checkpoint) {

    for (int i = 0; i < size; ++i) {
        int start = ids_trough[i];
        int end = ids_peak[i];
        if (X[start] < threshold && threshold < X[end]) {
            for (int j = 0; j < end - start - 1; ++j) {
                if (X[start + j] >= threshold) {
                    ids_checkpoint[i] = start + j;
                    break;
                }
            }
        }
    }
}

double threshold(double rate, double a, double b, double *A, double *B, double k, double W, int N) {
    int num_tcp = 1000;
    double step_tcp = 1.0 / (num_tcp - 1);
    double *list_tcp = malloc(num_tcp * sizeof(double)); // Ensure allocation
    double *list_output = malloc(num_tcp * sizeof(double)); // Ensure allocation
    double h;
    
    int id_min_value, id_max_value;
    for (int i = 0; i < num_tcp; i++) {
        list_tcp[i] = i * step_tcp;
        list_output[i] = output(list_tcp[i], a, b, A, B, k, W, N);
    }

    findMinMax(list_output, num_tcp, &id_min_value, &id_max_value);

    h = list_output[id_min_value] + rate * (list_output[id_max_value] - list_output[id_min_value]);

    free(list_tcp);
    free(list_output);

    return h;
}

double find_tcp(double h, double a, double b, double *A, double *B, double k, double W, int N) {
    int num_tcp = 10000;
    double step_tcp = 1.0 / (num_tcp - 1);
    double increment = 1.0/num_tcp;
    double current_output;
    double *list_tcp = malloc(num_tcp * sizeof(double)); // Ensure allocation
    double *list_output = malloc(num_tcp * sizeof(double)); // Ensure allocation
    double tcp;
    int id_min_value, id_max_value;
    int getout = 0;

    for (int i = 0; i < num_tcp; i++) {
        list_tcp[i] = i * step_tcp;
        list_output[i] = output(list_tcp[i], a, b, A, B, k, W, N);
    }

    findMinMax(list_output, num_tcp, &id_min_value, &id_max_value);
    tcp = list_tcp[id_min_value];
    double previous_output = output(tcp, a, b, A, B, k, W, N);  // Note: A and B are not used in this case

    if (previous_output == h && output(tcp-increment, a, b, A, B, k, W, N) < h) {
        getout = 1;
    }

    while (getout==0) {
        tcp += increment;
            
        current_output = output(tcp, a, b, A, B, k, W, N);

        if (previous_output < h && current_output >= h){
            free(list_tcp);
            free(list_output);
            return tcp;
            break;
        }

        previous_output = current_output;
    }  
}

void findMinMax(const double waveform[], int size, int *min_value, int *max_value) {
    *min_value = 0;
    *max_value = 0;

    for (int i = 1; i < size; ++i) {
        if (waveform[i] < waveform[*min_value]) {
            *min_value = i;
        }
        if (waveform[i] > waveform[*max_value]) {
            *max_value = i;
        }
    }
}

void normalise_vector(double *x, int dim) {
    double length = 0.0;

    for (int i = 0; i < dim; i++) {
        length += x[i] * x[i];
    }

    if (length > 0.0) {
        length = sqrt(length);
        for (int i = 0; i < dim; i++) {
            x[i] /= length;
        }
    }
}

double CV_ana(double* params, double T, double k, double W, double e, double D, int N, double tcp) {

    double A[N], B[N];

    for (int i = 0; i < N; i++) {
        A[i] = params[i];
        B[i] = params[N + i];
    }

    double f1 = func1(tcp, k, W, A, B, N);
    double f2 = func2(tcp, k, W, A, B, N);
    double f3 = func3(tcp, k, W, A, B, N);
    double R1 = D * T / pow(W, 2.0);
    double R2 = D * (1 - exp(-k * T)) / pow(f1, 2.0) * f2;
    double R3 = -D * (1 - exp(-k * T)) / (W * f1) * f3;
    double CV = e * sqrt(R1 + R2 + 2 * R3) / T;

    return CV;
}

double func1(double tcp, double k, double W, double *A, double *B, int N){

    double Sum, A_n, B_n, A_m, B_m, Phi_n, Phi_m;

    Sum = 0.0;

    int M = N;

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

double func2(double tcp, double k, double W, double *A, double *B, int N){

    double Sum, sum1, sum2, sum3, A_n, B_n, A_m, B_m, Phi_n, Phi_m;

    Sum = 0.0;

    int M = N;

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

double func3(double tcp, double k, double W, double *A, double *B, int N){

    double Sum, sum1, sum2, A_n, B_n, Phi_n;

    Sum = 0.0;

    int M = N;

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

void writedata_column(char *file, char *header, double *data, int length){

    FILE *fp = fopen(file, "w");

    //write header
    fprintf(fp,"%s\n",header);

    //write a set of data
    for(int i=0;i<length;i++){
        fprintf(fp,"%.15lf\n",data[i]);
    }
    fclose(fp);

    printf("write csv file: %s\n", file);
}

void writedata_row(char *file, char **header, double *data, int length){
    FILE *fp = fopen(file, "w");

    if (fp == NULL) {
        perror("can't open a csv file");
        return;
    }

    for (int i = 0; i < length; i++) {
        fprintf(fp, "%s", header[i]);
        if (i < length - 1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "\n");

    for (int i = 0; i < length; i++) {
        fprintf(fp, "%.15lf", data[i]);
        if (i < length - 1) {
            fprintf(fp, ",");
        }
    }
    fprintf(fp, "\n");

    fclose(fp);

    printf("write a csv file: %s\n", file);
}

void writedata_2d(char *file, char **header, double **data, int rows, int columns) {
    FILE *fp = fopen(file, "w");

    if (fp == NULL) {
        perror("can't open a csv file");
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

    printf("write a csv file: %s\n", file);
}

void readdata(const char *file, double *dataArray){
    FILE *fp = fopen(file, "r");

    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    char line[MAX_LINE_SIZE];

    int numElements = 0; // Use dereferencing to update the value pointed to by numElements

    // Skip the header (assuming it's the first line)
    if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return; // Error or empty file
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        char *token = strtok(line, ",");
        
        while (token != NULL) {
            dataArray[(numElements)++] = atof(token);
            token = strtok(NULL, ",");
        }
    }

    fclose(fp);
}
