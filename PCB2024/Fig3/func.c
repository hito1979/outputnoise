#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <limits.h>
#include <omp.h>
#include "SFMT-neon.h" //Mersenne Twister
#include "func.h"  // Include the header for function declarations
#define MAX_LINE_SIZE 1024

//**********************************************************************************************************************************************************************************
//Gibbs sampling
void gibbs_sampling(int flag_continue, int num_continue, int length, char **header, double **sample, double T_temp, int num_weight, int iteration, Bounds bounds_x, double a, double b, int N, double k_x, int step, double W, double dt, double e, double D, int start_time, double distance, int num_peak, int sample_period, int repetition, int seed_uniform, unsigned int *base_seed) {
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
        seed = *base_seed;
        CV(a, b, A, B, k_x, step, W, dt, e, D, N, num_peak, start_time, distance, sample_period, repetition, seed, &CV_x);

        sample[0][0] = CV_x;
        sample_index = 1; //the current parameter
        num = 1; //the iteration
        num_file = 1; //the next file
        printf("iteration=%d\n", 1);
    } else {
        //to restart from the last time
        char file_sample[200];
        sprintf(file_sample, "%s/Result/Sample/sample_%d.csv", base_path, num_continue);
        readdata_2d(file_sample, sample, length, 2*N+1);
        sample_index = 0;
        num = num_continue*length; //the number of sample gotten by the last time
        num_file = num_continue+1; //the file number will be producted next
        int start_dim = (num_continue * length ) % dim;
        start_dim = (start_dim == 0) ? dim : start_dim;
        start_dim = start_dim - 1;

        //to skip some random value to reproduct the last time continue
        int num_skip = dim + (num_continue*length - 1);
        for (int i = 0; i < num_skip; i++) {
            generator = sfmt_genrand_uint64(&sfmt);
            uniform_random = sfmt_to_res53(generator); //first uniform
        }
        //to skip some random seed to reproduct the last time continue
        //seed = *base_seed + num_weight*(num_continue)*(length-1)*repetition;
        *base_seed = (unsigned int)(num_weight*(num_continue*length-1)*repetition);
        double start = clock();
        for (int j = start_dim; j < dim; j++) {
            if (sample_index == 0){
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
            calculate_weights(num_weight, weights, new_CV, delta, T_temp, &sample[sample_index][1], j, bounds_x, a, b, N, k_x, step, W, dt, e, D, start_time, distance, num_peak, sample_period, repetition, base_seed);
            index = get_weighted_random_sample(num_weight, weights, delta, j, uniform_random, bounds_x);
            sample[sample_index][j+1] = bounds_x.lower_bound + index * delta;
            sample[sample_index][0] = new_CV[index];  //update CV

            // Save data every 'length' iterations
            if ((sample_index % (length-1) == 0) && (sample_index != 0)) {
                char file_sample[200];
                num++;
                printf("iteration=%d\n", num);
                sprintf(file_sample, "%s/Result/Sample/sample_%d.csv", base_path, num_file);
                writedata_2d(file_sample, header, sample, length, 2 * N + 1);
                double end = clock();
                double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
                printf("File %d generated in %lf seconds\n", num_file, cpu_time_used);
                num_file++; //to the next file
                sample_index = 0;
            }else{
                sample_index++;
                num++;
                printf("iteration=%d\n", num);
            }
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
            calculate_weights(num_weight, weights, new_CV, delta, T_temp, &sample[sample_index][1], j, bounds_x, a, b, N, k_x, step, W, dt, e, D, start_time, distance, num_peak, sample_period, repetition, base_seed);
            index = get_weighted_random_sample(num_weight, weights, delta, j, uniform_random, bounds_x);
            sample[sample_index][j+1] = bounds_x.lower_bound + (double)(index) * delta;
            sample[sample_index][0] = new_CV[index];  //update CV

            // Save data every 'length' iterations
            if ((sample_index % (length-1) == 0) && (sample_index != 0)) {
                char file_sample[200];
                num++;
                printf("iteration=%d\n", num);
                sprintf(file_sample, "%s/Result/Sample/sample_%d.csv", base_path, num_file);
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
        sprintf(file_sample, "%s/Result/sample_%d.csv", base_path, num_file);
        writedata_2d(file_sample, header, sample, sample_index, 2 * N + 1);
    }
}

void calculate_weights(int num_weight, double* weights, double *new_CV, double delta, double T_temp, double* current_params, int param_index, Bounds bounds_x, double a, double b, int N, double k_x, int step, double W, double dt, double e, double D, int start_time, double distance, int num_peak, int sample_period, int repetition, unsigned int *base_seed) {
    double range = bounds_x.upper_bound - bounds_x.lower_bound;
    double cv_value;
    int current_index;
    double A[N], B[N];
    unsigned int seed = *base_seed;

    for (int i = 0; i < N; i++) {
        //because current_params store only parameter without CV
        A[i] = current_params[i];
        B[i] = current_params[i + N];

    }
    if (param_index < N){
        current_index = param_index;

        for (int k = 0; k < num_weight; k++) {
            A[current_index] = bounds_x.lower_bound + k * delta;
            seed = seed + repetition;
            CV(a, b, A, B, k_x, step, W, dt, e, D, N, num_peak, start_time, distance, sample_period, repetition, seed, &cv_value);
            weights[k] = exp(-cv_value / T_temp);
            new_CV[k] = cv_value;
        }

    }else{
        current_index = param_index - N;

        for (int k = 0; k < num_weight; k++) {
            B[current_index] = bounds_x.lower_bound + k * delta;
            seed = seed + repetition;
            CV(a, b, A, B, k_x, step, W, dt, e, D, N, num_peak, start_time, distance, sample_period, repetition, seed, &cv_value);
            weights[k] = exp(-cv_value / T_temp);
            new_CV[k] = cv_value;
        }
    }  
    *base_seed = seed;
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

//**********************************************************************************************************************************************************************************
//Numerical CV

void CV(double a, double b, double *A, double *B, double k, int step, double W, double dt, double e, double D, int N, int count, int start_time, int distance, int sample_period, int repetition, unsigned int base_seed, double *CV_x) {
    //make array

    //#################################
    //reduction
    double cv_x = 0.0;

    //#################################
    //share
    int size = (int)(6.0/dt);
    //combo
    double combo1 = W * dt;
    double combo2 = e * sqrt(D*dt);

    //#################################
    //private
    int i, j, n;

    double theta, x;
    double next_theta, next_x;

    int min_period = distance; //the minimum distance between peaks
    int max_period = (int)(2.0 / dt) - distance; //the maximum distance between peaks
    double r; //random value
    double max_value_x; //local maximum or minimum
    int max_index_x; //index for local maximum or minimum in arr_x
    int index, index_x; //index for x to search peaks
    int error_x; //error flag
    int previous_peak_x; //index for previous peaks in x
    int index_peak_x; //index of the peak of x to search right now.

    double period_x[count+100];
    int ids_peak_x[count+100];
    double arr_x[size];

    for (int i = 0; i < count+100; i++) {
        period_x[i] = 0.0;
        ids_peak_x[i] = 0;
    }

    for (int i = 0; i < size; i++) {
        arr_x[i] = 0.0;
    }

    int count_x;

    double diff_x;

    double mean_x;
    double std_x;

    unsigned int seed;

    //Mersenne Twister
    sfmt_t sfmt;
    uint64_t generator;
    double uniform1, uniform2;

    #pragma omp parallel for reduction(+:cv_x) private(i, j, theta, x, next_theta, next_x, r, max_value_x, max_index_x, index, index_x, error_x, previous_peak_x, index_peak_x, period_x, ids_peak_x, arr_x, count_x, diff_x, mean_x, std_x, seed, sfmt, generator, uniform1, uniform2)
    for (n = 0; n < repetition; n++) {

        theta = 0.0;
        x = 0.0;

        error_x = 0; //error flag

        index_peak_x = 0;

        arr_x[0] = 0.0;

        mean_x = 0.0;
        std_x = 0.0;

        //seed
        seed = (unsigned int)(base_seed + n);
        sfmt_init_gen_rand(&sfmt, seed);
        
        //for loop for euler method and detect peaks, peak detection works at the same time as euler method.
        for (i = 0; i < start_time + (int)(5.0/dt); i++) {

            //euler method

            //generate gauss by Mersenne Twister
            generator = sfmt_genrand_uint64(&sfmt);
            uniform1 = sfmt_to_res53(generator); //first uniform
            generator = sfmt_genrand_uint64(&sfmt);
            uniform2 = sfmt_to_res53(generator); //second uniform
            r = gauss(uniform1, uniform2);

            next_theta = theta + combo1 + r * combo2;
            next_x = x + dt * (a + b * f(theta, A, B, N) - k * x);

            int adjusted_index = (i+1)%size;
            arr_x[adjusted_index] = next_x;

            theta = next_theta;
            x = next_x;

            index = i - 10;

            ///*
            if ((start_time <= index) && (index < start_time + (int)(5.0/dt)-11)) {

                // search the first peak for x
                if (start_time == index){
                    error_x = 0;
                    max_index_x = 0;
                    max_value_x = -INFINITY;//to avoid no peak(local maximum) being found

                    for (j = 0; j <= 20; j++){
                        if (arr_x[index%size] < arr_x[(index-10+j)%size]){
                            error_x = 1;
                        }
                    }

                    if (error_x == 0){
                        max_value_x = arr_x[index%size];
                        max_index_x = index;
                    }
                } else {
                    error_x = 0;

                    if (arr_x[index%size] > max_value_x) {
                        for (j = 0; j <= 20; j++){
                            // If a nearby value is larger, set the error flag to 1
                            if (arr_x[index%size] < arr_x[(index-10+j+size)%size]){
                                error_x = 1;
                            }
                        }

                        if (error_x == 0){
                            max_value_x = arr_x[index%size];
                            max_index_x = index;
                        }
                    }
                }
            }
        }

        ids_peak_x[index_peak_x] = max_index_x;

        previous_peak_x = max_index_x;

        index_x = ids_peak_x[0];

        error_x = 0;
        max_index_x = 0;
        max_value_x = -INFINITY; //to avoid that the first value is the largest in each search range

        //for loop for euler method and detect peaks, peak detection works at the same time as euler method.
        for (i = start_time + (int)(5.0/dt); i < step - 1; i++) {

            //euler method

            //generate gauss by Mersenne Twister
            generator = sfmt_genrand_uint64(&sfmt);
            uniform1 = sfmt_to_res53(generator); //first uniform
            generator = sfmt_genrand_uint64(&sfmt);
            uniform2 = sfmt_to_res53(generator); //second uniform
            r = gauss(uniform1, uniform2);

            next_theta = theta + combo1 + r * combo2;
            next_x = x + dt * (a + b * f(theta, A, B, N) - k * x);

            int adjusted_index = (i+1)%size;
            arr_x[adjusted_index] = next_x;

            theta = next_theta;
            x = next_x;

            //search peaks of x
            if ((previous_peak_x+min_period <= index_x) && (index_x <= previous_peak_x+max_period)) {
                if ((previous_peak_x+min_period) == index_x){
                    //printf("hotate!!!\n");
                    error_x = 0;
                    max_index_x = 0;
                    max_value_x = -INFINITY; //to avoid that the first value is the largest in each search range
                    for (int j = 0; j <= 20; j++){
                        int adjusted_index_x = (index_x - 10 + j + size) % size;
                        if (arr_x[index_x % size] < arr_x[adjusted_index_x]) {
                            error_x = 1;
                        }
                        /*
                        if (arr_w[index_w%size] < arr_w[(index_w-10+j+size)%size]){
                            error_w = 1;
                        }
                        */
                    }
                    if (error_x == 0){
                        max_value_x = arr_x[index_x%size];
                        max_index_x = index_x;
                    }
                } else {
                    error_x = 0;
                    if (arr_x[index_x%size] > max_value_x) {
                        for (int j = 0; j <= 20; j++){
                            int adjusted_index_x = (index_x - 10 + j + size) % size;
                            if (arr_x[index_x % size] < arr_x[adjusted_index_x]) {
                                error_x = 1;
                            }
                            /*
                            if (arr_w[index_w%size] < arr_w[(index_w-10+j+size)%size]){
                                error_w = 1;
                            }
                            */
                        }
                        if (error_x == 0){
                            max_value_x = arr_x[index_x%size];
                            max_index_x = index_x;
                        }
                    }
                }

                //check if for loop reaches the end of a range to search peaks
                if (index_x == previous_peak_x+max_period) {
                    if (max_index_x == 0){//if no peak are found in the range of x, the search range will be expanded. max_idex_w still being 0 means no peaks is found.
                        previous_peak_x = previous_peak_x + (max_period - min_period);
                        /*
                        #pragma omp critical
                        {
                            printf("previous_peak_x=%d\n", previous_peak_x);
                            printf("ringbuffer=%d\n", (index_x-10+size)%size);
                            for (int j = 0; j <= 20; j++){
                                printf("max_valu=%lf\n",  arr_x[(index_x-10+j+size)%size]);
                            }
                        }
                        */
                    } else {
                        index_peak_x++;
                        ids_peak_x[index_peak_x] =  max_index_x;
                        previous_peak_x = max_index_x;
                    }
                }
            }
            index_x++;

            ///*
            if ((count <= index_peak_x)){
                break;
            }
        }

        measure_period(ids_peak_x, count, sample_period, dt, period_x, distance);

        mean_x = 0.0;
        std_x = 0.0;

        for (i = 0; i < sample_period; i++) {
            mean_x += period_x[i];
        }

        mean_x /= sample_period;
        for (i = 0; i < sample_period; i++) {
            std_x += pow(period_x[i] - mean_x, 2);
        }

        std_x = sqrt(std_x / sample_period);

        cv_x = cv_x + std_x / mean_x;

    }
    *CV_x = cv_x/repetition;
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

void measure_period(int ids_peak[], int size, int sample_period, double dt, double *period, int distance) {
    int count = 0;
    double diff;
    
    for (int i = 0; i < size - 1; i++) {
        diff = (double)(ids_peak[i + 1] - ids_peak[i]) * dt;

        if ((distance * dt <= diff) && (diff <= (2.0 - distance * dt))) {
            period[count] = diff;
            //printf("period=%lf\n", diff);
            count++;
        }

        if (count >= sample_period) {
            //printf("count_getout=%d\n", count);
            break;
        }
    }
}

//**********************************************************************************************************************************************************************************
//Others

void normalize_samples(double **data, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        double norm = 0.0;
        for (int j = 1; j < columns; j++) {  // Skip the first column
            norm += data[i][j] * data[i][j];
        }
        norm = sqrt(norm);

        for (int j = 1; j < columns; j++) {  // Skip the first column
            if (norm != 0.0) {
                data[i][j] /= norm;
            }
        }
    }
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

