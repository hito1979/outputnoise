#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include "SFMT-neon.h" //Mersenne Twister
#include "func.h" // Include the header for function declarations
#define MAX_LINE_SIZE 1024

//####################################################################################################################################################################
void CV(double a, double b, double scale, double shift, int num_pow,  double T, double k_u, double k_v, double k_w, double k_x, int step, double dt, double e, double D, int count, int start_time, int distance, int sample_period, int repetetion, int base_seed, double *CV_w, double *CV_x) {
    //make array

    //#################################
    //reduction
    double cv_x = 0.0;
    double cv_w = 0.0;

    //#################################
    //share
    int size = (int)(6.0/dt);
    double combo = e*sqrt(D)*sqrt(dt); //to save calculation as much as possible

    //#################################
    //private
    int i, j, n;

    double u, v, w, x;
    double next_u, next_v, next_w, next_x;

    int min_period = distance; //the minimum distance between peaks
    int max_period = (int)(2.0 / dt) - distance; //the maximum distance between peaks
    double r; //random value
    double max_value_w, max_value_x; //local maximum or minimum
    int max_index_w, max_index_x; //index for local maximum or minimum in arr_w or arr_x
    int index, index_w, index_x; //index for w and x to search peaks
    int error_w, error_x; //error flag
    int previous_peak_w, previous_peak_x; //index for previous peaks in w and x
    int index_peak_x, index_peak_w; //index of the peak of x or w to search right now.

    double period_w[count+100], period_x[count+100];
    int ids_peak_w[count+100], ids_peak_x[count+100];
    double arr_w[size];
    double arr_x[size];

    for (int i = 0; i < count+100; i++) {
        period_w[i] = 0.0;
        period_x[i] = 0.0;
        ids_peak_w[i] = 0;
        ids_peak_x[i] = 0;
    }

    for (int i = 0; i < size; i++) {
        arr_w[i] = 0.0;
        arr_x[i] = 0.0;
    }

    int count_w, count_x;

    double diff_w, diff_x;

    double mean_w, mean_x;
    double std_w, std_x;

    int seed = 0;

    sfmt_t sfmt;
    uint64_t generator;
    double uniform1, uniform2;

    #pragma omp parallel for reduction(+:cv_w, cv_x) private(i, j, u, v, w, x, next_u, next_v, next_w, next_x, r, max_value_w, max_value_x, max_index_w, max_index_x, index, index_w, index_x, error_w, error_x, previous_peak_w, previous_peak_x, index_peak_x, index_peak_w, period_w, period_x, ids_peak_w, ids_peak_x, arr_w, arr_x, count_w, count_x, diff_w, diff_x, mean_w, mean_x, std_w, std_x, seed, sfmt, generator, uniform1, uniform2)
    for (n = 0; n < repetetion; n++) {

        u = 0.0;
        v = 0.0;
        w = 0.0;
        x = 0.0;

        error_w = 0;
        error_x = 0; //error flag

        index_peak_x = 0;
        index_peak_w = 0; //index of the peak of x or w to search right now.

        arr_w[0] = 0.0;
        arr_x[0] = 0.0;

        mean_w = 0.0;
        mean_x = 0.0;
        std_w = 0.0;
        std_x = 0.0;

        //seed
        seed = base_seed + n;
        sfmt_init_gen_rand(&sfmt, seed);
        
        //for loop for euler method and detect peaks, peak detection works at the same time as euler method.
        for (i = 0; i < start_time + (int)(5.0/dt); i++) {

            //generate gauss by Mersenne Twister
            generator = sfmt_genrand_uint64(&sfmt);
            uniform1 = sfmt_to_res53(generator); //first uniform
            generator = sfmt_genrand_uint64(&sfmt);
            uniform2 = sfmt_to_res53(generator); //second uniform
            r = gauss(uniform1, uniform2);
            //printf("seed=%lf\n", r);

            next_u = u + T * ((1 / (1 + pow(w, 10)) - k_u * u) * dt + r * combo);
            next_v = v + T * dt * (u - k_v * v);
            next_w = w + T * dt * (v - k_w * w);
            next_x = x + dt * (a + b * (scale * pow(w, num_pow) + shift) - k_x * x);

            int adjusted_index = (i+1)%size;
            arr_w[adjusted_index] = next_w;
            arr_x[adjusted_index] = next_x;

            u = next_u;
            v = next_v;
            w = next_w;
            x = next_x;

            index = i - 10;

            ///*
            if ((start_time <= index) && (index < start_time + (int)(5.0/dt)-11)) {
                // search the first peak for w
                if (start_time == index){ 
                    error_w = 0;
                    max_index_w = 0;
                    max_value_w = -INFINITY; //to avoid no peak(local maximum) being found

                    for (j = 0; j <= 20; j++){
                        if (arr_w[index%size] < arr_w[(index-10+j)%size]){
                            error_w = 1;
                        }
                    }

                    if (error_w == 0){
                        max_value_w = arr_w[index%size];
                        max_index_w = index;
                    }
                }else {
                    error_w = 0;
                        
                    if (arr_w[index%size] > max_value_w) {
                            
                        for (j = 0; j <= 20; j++){
                            if (arr_w[index%size] < arr_w[(index-10+j+size)%size]){
                                error_w = 1;
                            }
                        }

                        if (error_w == 0){
                            max_value_w = arr_w[index%size];
                            max_index_w = index;
                        }
                    }
                }

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
                            //printf("hotate!\n");
                        }
                    }
                }
            }
            //*/
        }

        ids_peak_w[index_peak_w] = max_index_w;
        ids_peak_x[index_peak_x] = max_index_x;

        previous_peak_w = max_index_w;
        previous_peak_x = max_index_x;

        //printf("previous_x%d\n", previous_peak_x);

        index_w = ids_peak_w[0];
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
            r = gauss(uniform1, uniform2);

            //r = gauss();
            next_u = u + T * ((1 / (1 + pow(w, 10)) - k_u * u) * dt + r * combo);
            next_v = v + T * dt * (u - k_v * v);
            next_w = w + T * dt * (v - k_w * w);
            next_x = x + dt * (a + b * (scale * pow(w, num_pow) + shift) - k_x * x);

            //printf("diff=%lf\n", next_w-arr_x[i%size]);

            int adjusted_index = (i+1)%size;
            arr_w[adjusted_index] = next_w;
            arr_x[adjusted_index] = next_x;

            u = next_u;
            v = next_v;
            w = next_w;
            x = next_x;

            //search peaks of w
            if ((previous_peak_w+min_period <= index_w) && (index_w <= previous_peak_w+max_period)) {
                if ((previous_peak_w+min_period) == index_w){
                    error_w = 0;
                    max_index_w = 0;
                    max_value_w = -INFINITY;//to avoid no peak(local maximum) being found

                    for (int j = 0; j <= 20; j++){
                        int adjusted_index_w = (index_w - 10 + j + size) % size;
                        if (arr_w[index_w % size] < arr_w[adjusted_index_w]) {
                            error_w = 1;
                        }
                    }
                    if (error_w == 0){
                        max_value_w = arr_w[index_w%size];
                        max_index_w = index_w;
                    }
                } else {
                    error_w = 0;
                    if (arr_w[index_w%size] > max_value_w) {
                        for (int j = 0; j <= 20; j++){
                            int adjusted_index_w = (index_w - 10 + j + size) % size;
                            if (arr_w[index_w % size] < arr_w[adjusted_index_w]) {
                                error_w = 1;
                            }
                        }
                        if (error_w == 0){
                            max_value_w = arr_w[index_w%size];
                            max_index_w = index_w;
                        }
                    }
                }

                //check if for loop reaches the end of a range to search peaks
                if (index_w == previous_peak_w+max_period) {
                    if (max_index_w == 0){//if no peak are found in the range of x, the search range will be expanded. max_idex_w still being 0 means no peaks is found.
                        previous_peak_w = previous_peak_w + (max_period - min_period);
                    } else {
                        index_peak_w++;
                        ids_peak_w[index_peak_w] =  max_index_w;
                        previous_peak_w = max_index_w;
                    }
                }
            }
            index_w++;
        
            //search peaks of x
            if ((previous_peak_x+min_period <= index_x) && (index_x <= previous_peak_x+max_period)) {
                if ((previous_peak_x+min_period) == index_x){
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

            if ((count <= index_peak_w) && (count <= index_peak_x)){
                break;
            }
        }

        measure_period(ids_peak_w, count, sample_period, dt, period_w, distance);
        measure_period(ids_peak_x, count, sample_period, dt, period_x, distance);

        mean_w = 0.0; 
        mean_x = 0.0;
        std_w = 0.0;
        std_x = 0.0;

        for (i = 0; i < sample_period; i++) {
            mean_w += period_w[i];
            mean_x += period_x[i];
        }
        
        mean_w /= sample_period;
        mean_x /= sample_period;
        for (i = 0; i < sample_period; i++) {
            std_w += pow(period_w[i] - mean_w, 2);
            std_x += pow(period_x[i] - mean_x, 2);
        }

        std_w = sqrt(std_w / sample_period);
        std_x = sqrt(std_x / sample_period);

        cv_w = cv_w + std_w / mean_w;
        cv_x = cv_x + std_x / mean_x;

    }

    *CV_w = cv_w/repetetion;
    *CV_x = cv_x/repetetion;
}

double gauss(double uniform1, double uniform2) {

    double z = sqrt(-2 * log(uniform1)) * cos(2 * M_PI * uniform2);
    return z;
}

void measure_period(int ids_peak[], int size, int sample_period, double dt, double *period, int distance) {
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
