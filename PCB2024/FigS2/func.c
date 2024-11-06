// functions.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "SFMT.h" //Mersenne Twister
#include "func.h"  // Include the header for function declarations

//**********************************************************************************************************************************************************************************
//Differential_evolution

//DE
void differential_evolution(ObjectiveFunction objective, Normalise normalise, Bounds bounds_x, int dim, DEParams params, int iteration, char *stra, int popsize, double mutation, double crossover, double threshold_convergence, int count_convergence, double a, double b, double k, int step, double W, double dt, double e, double D, int N, int num_sample, int start_time, int distance, int sample_period, int repetition, int length, int seed){

    // Assuming params.population_size and dim are known constants
    typedef struct {
        double vector[dim];  // Statically allocate the vector array
        double fitness;
    } Individual;

    // Define the population statically
    Individual population[params.population_size];  // Statically allocated array of individuals
    //Individual *population = (Individual *)malloc(params.population_size * sizeof(Individual));  // Static array for population
    double mutant_vector[dim];                      // Static array for mutant vector
    double crossover_vector[dim];                   // Static array for crossover vector
    double best_solution[dim];                      // Static array for best solution
    double generation[length][2*N+1];
    char header[2*N+2][20]; // 2*N + 2 rows, each with 20 characters
    double initial_value[2*N+1];
    double best_individual[2*N+3]; //最終的な最適化されたパラメーターの保存
    double generation_first[popsize][2*N+1];  // Initial individual
    double generation_half[popsize][2*N+1];   // Progress individuals
    double generation_final[popsize][2*N+1];  // Final individual
    int best_generation = -1;
    int flag_convergence = 0;
    int count = 0;
    int num_file = 1;
    double best_fitness;
    char base_path[200] = "..";
    char file[200];
    int index;
    int gen_finish;
    double diff;
    unsigned int base_seed = seed;
    char file_generation[200], file_best_individual[200], file_generation_first[200], file_generation_half[200], file_generation_final[200];
    FILE *fp;

    // Define the header
    for (int i = 0; i < 2*N+2; i++) {
        if (i == 0) {
            strcpy(header[i], "flag_convergence");
        } else if (i == 1) {
            strcpy(header[i], "fitness");
        } else if (1 < i && i <= N + 1) {
            sprintf(header[i], "A_%d", i - 1);  // Directly format the string into the header
        } else {
            sprintf(header[i], "B_%d", i - N - 1);  // Directly format the string into the header
        }
    }

    //Mersenne Twister
    sfmt_t sfmt;
    sfmt_init_gen_rand(&sfmt, seed);
    uint64_t generator;
    double uniform_random;

    //initial value
    for (int i = 0; i < params.population_size; i++) {
        //population[i].vector = (double *)malloc(dim * sizeof(double));

        for (int j = 0; j < dim; j++) {

            //generate gauss by Mersenne Twister
            generator = sfmt_genrand_uint64(&sfmt);
            uniform_random = sfmt_to_res53(generator); //first uniform
            population[i].vector[j] = bounds_x.lower_bound + uniform_random * (bounds_x.upper_bound - bounds_x.lower_bound);
            generation_first[i][j+1] = population[i].vector[j];

        }
        
        normalise(population[i].vector, dim); //A_nとB_n
        for (int j = 0; j < dim; j++) {
            generation_first[i][j+1] = population[i].vector[j];
        }

        population[i].fitness = objective(a, b, population[i].vector, k, step, W, dt, e, D, N, num_sample, start_time, distance, sample_period, repetition, base_seed);
        base_seed = base_seed + (unsigned)(repetition);
        generation_first[i][0] = population[i].fitness;

    }

    for (int i = 0; i < params.population_size; i++){
        //CVの保存
        generation_half[i][0] = population[i].fitness;
        for (int j = 0; j < dim; j++) {
            generation_half[i][j+1] = population[i].vector[j];
        }
    }

    best_fitness = INFINITY;
    for (int i = 0; i < params.population_size; i++) {
        if (population[i].fitness < best_fitness) {
            best_fitness = population[i].fitness;
            for (int j = 0; j < dim; j++) {
                best_solution[j] = population[i].vector[j];
            }
        }
    }

    generation[0][0] = best_fitness;
    for (int i = 0; i < dim; i++) {
        generation[0][i+1] = best_solution[i];
    }
    int gen = 1;

    double previous_fitness = generation[0][0];

    while (gen <= params.max_generations && flag_convergence == 0) {

        for (int i = 0; i < params.population_size; i++) {
            int r1, r2, r3;
            do {
                generator = sfmt_genrand_uint64(&sfmt);
                r1 = generator % params.population_size;
            } while (r1 == i);
            do {
                generator = sfmt_genrand_uint64(&sfmt);
                r2 = generator % params.population_size;
            } while (r2 == i || r2 == r1);
            do {
                generator = sfmt_genrand_uint64(&sfmt);
                r3 = generator % params.population_size;
            } while (r3 == i || r3 == r1 || r3 == r2);

            for (int j = 0; j < dim; j++) {
                if (strcmp(params.strategy, "rand1bin") == 0) {
                    mutant_vector[j] = population[r1].vector[j] + params.mutation_factor * (population[r2].vector[j] - population[r3].vector[j]);
                } else {
                    //other strategy
                }
            }
            
            //crossoever
            generator = sfmt_genrand_uint64(&sfmt);
            int ri = generator % dim;
            for (int j = 0; j < dim; j++) {
                generator = sfmt_genrand_uint64(&sfmt);
                uniform_random = sfmt_to_res53(generator); //first uniform
                if (j == ri || (uniform_random < params.crossover_rate)) {
                    crossover_vector[j] = mutant_vector[j];
                } else {
                    crossover_vector[j] = population[i].vector[j];
                }
            }

            //boundary 
            normalise(crossover_vector, dim); //A_n, B_n

            //選択の操作
            double new_fitness = objective(a, b, crossover_vector, k, step, W, dt, e, D, N, num_sample, start_time, distance, sample_period, repetition, base_seed);
            base_seed = base_seed + (unsigned)(repetition);
            if (new_fitness < population[i].fitness) {
                for (int j = 0; j < dim; j++) {
                    population[i].vector[j] = crossover_vector[j];
                }
                population[i].fitness = new_fitness;
            }
        }

        //最良解の更新
        for (int i = 0; i < params.population_size; i++) {
            if (population[i].fitness < best_fitness) {
                best_fitness = population[i].fitness;
                for (int j = 0; j < dim; j++) {
                    best_solution[j] = population[i].vector[j];
                }
            }
        }

        //save optimal parameter
        index = gen%length;
        generation[index][0] = best_fitness;
        for (int i = 0; i < dim; i++) {
            generation[index][i+1] = best_solution[i];
        }

        //半分経過時の個体群の保存
        if (gen == params.population_size / 2){
            for (int i = 0; i < params.population_size; i++){
                //CVの保存
                generation_half[i][0] = population[i].fitness;
                for (int j = 0; j < dim; j++) {
                    //AnとBnの保存
                    generation_half[i][j+1] = population[i].vector[j];
                }
            }

        }

        //save generation
        if (index == (length-1)){
            //write a csv file
            sprintf(file_generation, "%s%s%d%s%d%s%d%s", base_path, "/Result/N=", N, "/", seed, "/Generation/generation_", num_file, ".csv");
            fp = fopen(file_generation, "w");
            if (fp == NULL) {
                perror("can't open a csv file");
            }else{
                for (int i = 0; i < 2*N+1; i++) {
                    fprintf(fp, "%s", header[i+1]);
                    if (i < (2*N+1) - 1) {
                        fprintf(fp, ",");
                    }
                }
                fprintf(fp, "\n");
                for (int i = 0; i < length; i++) {
                    for (int j = 0; j < 2*N+1; j++) {
                        fprintf(fp, "%.15lf", generation[i][j]);
                        if (j < (2*N+1) - 1) {
                            fprintf(fp, ",");
                        }
                    }
                    fprintf(fp, "\n");
                }
                fclose(fp);
                printf("complete writing a csv file: %s\n", file_generation);
            }

            num_file++;
        }
        printf("Generation: %d, Best Fitness: %0.15lf\n", gen, best_fitness);

        diff = previous_fitness-generation[index][0];
        printf("previous=%lf\n", previous_fitness);
        printf("now=%lf\n", generation[index][0]);

        if (threshold_convergence > diff){
            count++;
            printf("hotate\n");
            if (count_convergence <= count){
                flag_convergence = 1;
                gen_finish = gen;
            }
        }
        else{
            count = 0;
        }
        previous_fitness = generation[index][0];


        gen++;

    }

    gen_finish = gen-1;

    //final population
    for (int i = 0; i < params.population_size; i++){
        generation_final[i][0] = population[i].fitness;
        for (int j = 0; j < dim; j++) {
            generation_final[i][j+1] = population[i].vector[j];
        }
    }

    best_individual[0] = flag_convergence;
    best_individual[1] = best_fitness;
    for (int i = 0; i < dim; i++) {
        best_individual[i+2] = best_solution[i];
    }

    index = gen_finish%length;
    if (index != (length-1)) {
        sprintf(file_generation, "%s%s%d%s%d%s%d%s", base_path, "/Result/N=", N, "/", seed, "/Generation/generation_", num_file, ".csv");
        fp = fopen(file_generation, "w");
        if (fp == NULL) {
            perror("can't open a csv file");
        }else{
            for (int i = 0; i < 2*N+1; i++) {
                fprintf(fp, "%s", header[i+1]);
                if (i < (2*N+1) - 1) {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, "\n");
            for (int i = 0; i < index+1; i++) {
                for (int j = 0; j < 2*N+1; j++) {
                    fprintf(fp, "%.15lf", generation[i][j]);
                    if (j < (2*N+1) - 1) {
                        fprintf(fp, ",");
                    }
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
            printf("complete writing a csv file: %s\n", file_generation);
        }
    }

    //write a csv file for best_individual
    sprintf(file_best_individual, "%s%s%d%s%d%s", base_path, "/Result/N=", N, "/", seed, "/Generation/best_individual.csv");
    fp = fopen(file_best_individual, "w");
    if (fp == NULL) {
        perror("can't open a file");
        return;
    }else{
        for (int i = 0; i < 2*N+2; i++) {
            fprintf(fp, "%s", header[i]);
            if (i < (2*N+2) - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
        for (int i = 0; i < 2*N+2; i++) {
            fprintf(fp, "%.15lf", best_individual[i]);
            if (i < (2*N+2) - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
        fclose(fp);
        printf("complete writing a csv file: %s\n", file_best_individual);
    }


    //write a csv file for generation_first
    sprintf(file_generation_first, "%s%s%d%s%d%s", base_path, "/Result/N=", N, "/", seed, "/Generation/generation_first.csv");
    fp = fopen(file_generation_first, "w");
    if (fp == NULL) {
        perror("can't open a csv file");
    }else{
        for (int i = 0; i < 2*N+1; i++) {
            fprintf(fp, "%s", header[i+1]);
            if (i < (2*N+1) - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
        for (int i = 0; i < popsize; i++) {
            for (int j = 0; j < 2*N+1; j++) {
                fprintf(fp, "%.15lf", generation_first[i][j]);
                if (j < (2*N+1) - 1) {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("complete writing a csv file: %s\n", file_generation_first);
    }

    //write a csv file for generation_half
    sprintf(file_generation_half, "%s%s%d%s%d%s", base_path, "/Result/N=", N, "/", seed, "/Generation/generation_half.csv");
    fp = fopen(file_generation_half, "w");
    if (fp == NULL) {
        perror("can't open a csv file");
    }else{
        for (int i = 0; i < 2*N+1; i++) {
            fprintf(fp, "%s", header[i+1]);
            if (i < (2*N+1) - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
        for (int i = 0; i < popsize; i++) {
            for (int j = 0; j < 2*N+1; j++) {
                fprintf(fp, "%.15lf", generation_half[i][j]);
                if (j < (2*N+1) - 1) {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("complete writing a csv file: %s\n", file_generation_half);
    }

    //write a csv file for generation_final
    sprintf(file_generation_final, "%s%s%d%s%d%s", base_path, "/Result/N=", N, "/", seed, "/Generation/generation_final.csv");
    fp = fopen(file_generation_final, "w");
    if (fp == NULL) {
        perror("can't open a csv file");
    }else{
        for (int i = 0; i < 2*N+1; i++) {
            fprintf(fp, "%s", header[i+1]);
            if (i < (2*N+1) - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
        for (int i = 0; i < popsize; i++) {
            for (int j = 0; j < 2*N+1; j++) {
                fprintf(fp, "%.15lf", generation_final[i][j]);
                if (j < (2*N+1) - 1) {
                    fprintf(fp, ",");
                }
            }
            fprintf(fp, "\n");
        }
        fclose(fp);
        printf("complete writing a csv file: %s\n", file_generation_final);
    }
}

void normalise_vector(double *x, int dim) {
    double length = 0.0;

    //norm
    for (int i = 0; i < dim; i++) {
        length += x[i] * x[i];
    }

    //each elements divided by norm
    if (length > 0.0) {
        length = sqrt(length);
        for (int i = 0; i < dim; i++) {
            x[i] /= length;
        }
    }
}

//****************************************************************************************
//objective function
double CV(double a, double b, double *params, double k, int step, double W, double dt, double e, double D, int N, int num_sample, int start_time, int distance, int sample_period, int repetition, unsigned int base_seed) {
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
    double A[N], B[N];
    for (int i = 0; i < N; i++) {
        A[i] = params[i];
    }
    for (int i = 0; i < N; i++) {
        B[i] = params[N+i];
    }

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

    double period_x[num_sample+100];
    int ids_peak_x[num_sample+100];
    double arr_x[size];

    for (int i = 0; i < num_sample+100; i++) {
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
                    } else {
                        index_peak_x++;
                        ids_peak_x[index_peak_x] =  max_index_x;
                        previous_peak_x = max_index_x;
                    }
                }
            }
            index_x++;

            ///*
            if ((num_sample <= index_peak_x)){
                break;
            }
        }

        measure_period(ids_peak_x, num_sample, sample_period, dt, period_x, distance);

        mean_x = 0.0;
        std_x = 0.0;

        for (i = 0; i < sample_period; i++) {
            mean_x += period_x[i];
            //printf("period=%lf\n", period_x[i]);
        }

        mean_x /= sample_period;
        for (i = 0; i < sample_period; i++) {
            std_x += pow(period_x[i] - mean_x, 2);
        }

        std_x = sqrt(std_x / sample_period);

        cv_x = cv_x + std_x / mean_x;

        //printf("cv=%lf\n", std_x / mean_x);

    }
    double CV_x = cv_x/repetition;

    //printf("cv_x=%lf\n", cv_x);
    //printf("CV_x=%lf\n", CV_x);

    return CV_x;
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

void measure_period(int *ids_peak, int size, int sample_period, double dt, double *period, int distance) {
    int count = 0;
    double diff;
    
    for (int i = 0; i < size - 1; i++) {
        diff = (double)(ids_peak[i + 1] - ids_peak[i]) * dt;

        if ((distance * dt <= diff) && (diff <= (2.0 - distance * dt))) {
            period[count] = diff;
            count++;
        }

        if (count >= sample_period) {
            break;
        }
    }
}

