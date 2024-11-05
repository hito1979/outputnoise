// functions.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "SFMT-neon.h" //Mersenne Twister
#include "func.h"  // Include the header for function declarations

//**********************************************************************************************************************************************************************************
//Differential_evolution

//DE
void differential_evolution(ObjectiveFunction objective, Normalise normalise, Bounds bounds_x, int dim, DEParams params, int iteration, char *stra, int popsize, double mutation, double crossover, double threshold_convergence, int count_convergence, int N, double W, double T, double e, double D, double a, double b, int num_tcp, double *list_tcp, int num_k, double *list_k, int num_rate, double *list_rate, int length, int seed){

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

        population[i].fitness = objective(population[i].vector, N, W, T, e, D, a, b, num_tcp, list_tcp, num_k, list_k, num_rate, list_rate);
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
            double new_fitness = objective(crossover_vector, N, W, T, e, D, a, b, num_tcp, list_tcp, num_k, list_k, num_rate, list_rate);
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

        diff = previous_fitness - generation[index][0];
        if (threshold_convergence > diff){
            count++;
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

    gen_finish = gen;

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
    sprintf(file_generation_final, "%s%s%d%s%d%s", base_path, "/Result/N=", N, "/generation_final_", seed, ".csv");
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

    //each element divided by norm
    if (length > 0.0) {
        length = sqrt(length);
        for (int i = 0; i < dim; i++) {
            x[i] /= length;
        }
    }
}

//****************************************************************************************
//objective function
double Func(double *params, int N, double W, double T, double e, double D, double a, double b, int num_tcp, double *list_tcp, int num_k, double *list_k, int num_rate, double *list_rate){

    double A[N], B[N];
    double list_output[num_k][num_tcp];
    double tcp, rate, k, h, R1, R2, R3, f1, f2, f3;
    double CV = 0.0;
    double cv = 0.0;
    int i, j, l, id_min, id_max, id_tcp;

    for (int i = 0; i < N; i++) {
        A[i] = params[i];
    }
    for (int i = 0; i < N; i++) {
        B[i] = params[N+i];
    }

    //determine threshold
    for (i = 0; i < num_k; i++){
        k = list_k[i];
        #pragma omp parallel for private(tcp)
        for (j = 0; j < num_tcp; j++) {
            tcp = list_tcp[j];
            list_output[i][j] = output(tcp, k, A, B, N, W, a, b);
        }
    }

    #pragma omp parallel for reduction(+:CV) private(i, j, k, l, id_min, id_max, rate, h, id_tcp, f1, f2, f3, R1, R2, R3) collapse(2)
    for (i = 0; i < num_k; i++){
        for (j = 0; j < num_rate; j++){

            k = list_k[i];

            findMinMax(&(list_output[i][0]), num_tcp, &id_min, &id_max);

            rate = list_rate[j];
            h = list_output[i][id_min] + rate*(list_output[i][id_max]-list_output[i][id_min]);

            id_tcp = find_tcp(&(list_output[i][0]), num_tcp, list_tcp, h);
            f1 = func1(list_tcp[id_tcp], k, A, B, N, W);
            f2 = func2(list_tcp[id_tcp], k, A, B, N, W);
            f3 = func3(list_tcp[id_tcp], k, A, B, N, W);
            R1 = D*T/pow(W, 2.0);
            R2 = D*(1-exp(-k*T))/pow(f1,2)*f2;
            R3 = -D*(1-exp(-k*T))/(W*f1)*f3;
            CV = CV + e*sqrt(R1+R2+2*R3)/T;
            
        }

    }


    CV = CV/(num_k*num_rate);

    return CV;
}

double func1(double tcp, double k, double *A, double *B, int N, double W){

    double Sum, A_n, B_n, A_m, B_m, Phi_n, Phi_m;
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

double func2(double tcp, double k, double *A, double *B, int N, double W){

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

double func3(double tcp, double k, double *A, double *B, int N, double W){

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

void findMinMax(const double waveform[], int size, int *id_min, int *id_max) {
    *id_min = 0;
    *id_max = 0;

    for (int i = 1; i < size; ++i) {
        if (waveform[i] < waveform[*id_min]) {
            *id_min = i;
        }
        if (waveform[i] > waveform[*id_max]) {
            *id_max = i;
        }
    }
}

int find_tcp(double *list_output, int num_tcp, double *list_tcp, double h){
    int id_tcp = 0;
    double previous_output = list_output[id_tcp];  // Note: A and B are not used in this case
    double current_output;

    // serach check point

    if (previous_output == h) {
        return id_tcp;
    }

    for (int i = 0; i < num_tcp; i++) {
        id_tcp = id_tcp + 1;
        current_output = list_output[id_tcp];
        if (previous_output < h && current_output >= h){
            return id_tcp;
        }
        previous_output = current_output;
    }
            
    printf("error!!!\n");
    return -1;
}

double output(double t, double k, double *A, double *B, int N, double W, double a, double b) {
    double Sum = a / k;

    for (int n = 1; n <= N; ++n) {
        Sum += b * 1 / (pow(k, 2) + pow(n*W, 2)) * ((n*W*A[n-1] + k*B[n-1]) * sin(n*W*t) + (k*A[n-1] - n*W*B[n-1]) * cos(n*W*t));
    }

    return Sum;
}
