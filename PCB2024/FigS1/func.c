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
void CV_u(double a, double b, double scale, double shift, double num_pow,  double T, double k_u, double k_v, double k_w, double k_x, int step, double dt, double e, double D, int count, int start_time, int distance, int sample_period, int repetetion, int base_seed, double *CV_u, double *CV_x) {
    //make array

    //#################################
    //reduction
    double cv_x = 0.0;
    double cv_u = 0.0;

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
    double max_value_u, max_value_x; //local maximum or minimum
    int max_index_u, max_index_x; //index for local maximum or minimum in arr_w or arr_x
    int index, index_u, index_x; //index for w and x to search peaks
    int error_u, error_x; //error flag
    int previous_peak_u, previous_peak_x; //index for previous peaks in w and x
    int index_peak_x, index_peak_u; //index of the peak of x or w to search right now.

    double period_u[count+100], period_x[count+100];
    int ids_peak_u[count+100], ids_peak_x[count+100];
    double arr_u[size];
    double arr_x[size];

    for (int i = 0; i < count+100; i++) {
        period_u[i] = 0.0;
        period_x[i] = 0.0;
        ids_peak_u[i] = 0;
        ids_peak_x[i] = 0;
    }

    for (int i = 0; i < size; i++) {
        arr_u[i] = 0.0;
        arr_x[i] = 0.0;
    }

    int count_u, count_x;

    double diff_u, diff_x;

    double mean_u, mean_x;
    double std_u, std_x;

    int seed = 0;

    sfmt_t sfmt;
    uint64_t generator;
    double uniform1, uniform2;

    #pragma omp parallel for reduction(+:cv_u, cv_x) private(i, j, u, v, w, x, next_u, next_v, next_w, next_x, r, max_value_u, max_value_x, max_index_u, max_index_x, index, index_u, index_x, error_u, error_x, previous_peak_u, previous_peak_x, index_peak_x, index_peak_u, period_u, period_x, ids_peak_u, ids_peak_x, arr_u, arr_x, count_u, count_x, diff_u, diff_x, mean_u, mean_x, std_u, std_x, seed, sfmt, generator, uniform1, uniform2)
    for (n = 0; n < repetetion; n++) {

        u = 0.0;
        v = 0.0;
        w = 0.0;
        x = 0.0;

        error_u = 0;
        error_x = 0; //error flag

        index_peak_x = 0;
        index_peak_u = 0; //index of the peak of x or w to search right now.

        arr_u[0] = 0.0;
        arr_x[0] = 0.0;

        mean_u = 0.0;
        mean_x = 0.0;
        std_u = 0.0;
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
            next_x = x + dt * (a + b * (scale * pow(u, num_pow) + shift) - k_x * x);

            int adjusted_index = (i+1)%size;
            arr_u[adjusted_index] = next_u;
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
                    error_u = 0;
                    max_index_u = 0;
                    max_value_u = -INFINITY; //to avoid no peak(local maximum) being found

                    for (j = 0; j <= 20; j++){
                        if (arr_u[index%size] < arr_u[(index-10+j)%size]){
                            error_u = 1;
                        }
                    }

                    if (error_u == 0){
                        max_value_u = arr_u[index%size];
                        max_index_u = index;
                    }
                }else {
                    error_u = 0;
                        
                    if (arr_u[index%size] > max_value_u) {
                            
                        for (j = 0; j <= 20; j++){
                            if (arr_u[index%size] < arr_u[(index-10+j+size)%size]){
                                error_u = 1;
                            }
                        }

                        if (error_u == 0){
                            max_value_u = arr_u[index%size];
                            max_index_u = index;
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

        ids_peak_u[index_peak_u] = max_index_u;
        ids_peak_x[index_peak_x] = max_index_x;

        previous_peak_u = max_index_u;
        previous_peak_x = max_index_x;

        //printf("previous_x%d\n", previous_peak_x);

        index_u = ids_peak_u[0];
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
            next_x = x + dt * (a + b * (scale * pow(u, num_pow) + shift) - k_x * x);

            //printf("diff=%lf\n", next_w-arr_x[i%size]);

            int adjusted_index = (i+1)%size;
            arr_u[adjusted_index] = next_u;
            arr_x[adjusted_index] = next_x;

            u = next_u;
            v = next_v;
            w = next_w;
            x = next_x;

            //search peaks of w
            if ((previous_peak_u+min_period <= index_u) && (index_u <= previous_peak_u+max_period)) {
                if ((previous_peak_u+min_period) == index_u){
                    error_u = 0;
                    max_index_u = 0;
                    max_value_u = -INFINITY;//to avoid no peak(local maximum) being found

                    for (int j = 0; j <= 20; j++){
                        int adjusted_index_u = (index_u - 10 + j + size) % size;
                        if (arr_u[index_u % size] < arr_u[adjusted_index_u]) {
                            error_u = 1;
                        }
                    }
                    if (error_u == 0){
                        max_value_u = arr_u[index_u%size];
                        max_index_u = index_u;
                    }
                } else {
                    error_u = 0;
                    if (arr_u[index_u%size] > max_value_u) {
                        for (int j = 0; j <= 20; j++){
                            int adjusted_index_u = (index_u - 10 + j + size) % size;
                            if (arr_u[index_u % size] < arr_u[adjusted_index_u]) {
                                error_u = 1;
                            }
                        }
                        if (error_u == 0){
                            max_value_u = arr_u[index_u%size];
                            max_index_u = index_u;
                        }
                    }
                }

                //check if for loop reaches the end of a range to search peaks
                if (index_u == previous_peak_u+max_period) {
                    if (max_index_u == 0){//if no peak are found in the range of x, the search range will be expanded. max_idex_w still being 0 means no peaks is found.
                        previous_peak_u = previous_peak_u + (max_period - min_period);
                    } else {
                        index_peak_u++;
                        ids_peak_u[index_peak_u] =  max_index_u;
                        previous_peak_u = max_index_u;
                    }
                }
            }
            index_u++;
        
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
            if ((count <= index_peak_u) && (count <= index_peak_x)){
                //printf("hotateeeeeeeeeeeeeeeeee\n");
                break;
            }
        }

        measure_period(ids_peak_u, count, sample_period, dt, period_u, distance);
        measure_period(ids_peak_x, count, sample_period, dt, period_x, distance);

        mean_u = 0.0; 
        mean_x = 0.0;
        std_u = 0.0;
        std_x = 0.0;

        for (i = 0; i < sample_period; i++) {
            mean_u += period_u[i];
            mean_x += period_x[i];
        }
        
        mean_u /= sample_period;
        mean_x /= sample_period;
        for (i = 0; i < sample_period; i++) {
            std_u += pow(period_u[i] - mean_u, 2);
            std_x += pow(period_x[i] - mean_x, 2);
        }

        std_u = sqrt(std_u / sample_period);
        std_x = sqrt(std_x / sample_period);

        cv_u = cv_u + std_u / mean_u;
        cv_x = cv_x + std_x / mean_x;

    }

    *CV_u = cv_u/repetetion;
    *CV_x = cv_x/repetetion;
}

void CV_v(double a, double b, double scale, double shift, double num_pow,  double T, double k_u, double k_v, double k_w, double k_x, int step, double dt, double e, double D, int count, int start_time, int distance, int sample_period, int repetetion, int base_seed, double *CV_v, double *CV_x) {
    //make array

    //#################################
    //reduction
    double cv_x = 0.0;
    double cv_v = 0.0;

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
    double max_value_v, max_value_x; //local maximum or minimum
    int max_index_v, max_index_x; //index for local maximum or minimum in arr_w or arr_x
    int index, index_v, index_x; //index for w and x to search peaks
    int error_v, error_x; //error flag
    int previous_peak_v, previous_peak_x; //index for previous peaks in w and x
    int index_peak_x, index_peak_v; //index of the peak of x or w to search right now.

    double period_v[count+100], period_x[count+100];
    int ids_peak_v[count+100], ids_peak_x[count+100];
    double arr_v[size];
    double arr_x[size];

    for (int i = 0; i < count+100; i++) {
        period_v[i] = 0.0;
        period_x[i] = 0.0;
        ids_peak_v[i] = 0;
        ids_peak_x[i] = 0;
    }

    for (int i = 0; i < size; i++) {
        arr_v[i] = 0.0;
        arr_x[i] = 0.0;
    }

    int count_v, count_x;

    double diff_v, diff_x;

    double mean_v, mean_x;
    double std_v, std_x;

    int seed = 0;

    sfmt_t sfmt;
    uint64_t generator;
    double uniform1, uniform2;

    #pragma omp parallel for reduction(+:cv_v, cv_x) private(i, j, u, v, w, x, next_u, next_v, next_w, next_x, r, max_value_v, max_value_x, max_index_v, max_index_x, index, index_v, index_x, error_v, error_x, previous_peak_v, previous_peak_x, index_peak_x, index_peak_v, period_v, period_x, ids_peak_v, ids_peak_x, arr_v, arr_x, count_v, count_x, diff_v, diff_x, mean_v, mean_x, std_v, std_x, seed, sfmt, generator, uniform1, uniform2)
    for (n = 0; n < repetetion; n++) {

        u = 0.0;
        v = 0.0;
        w = 0.0;
        x = 0.0;

        error_v = 0;
        error_x = 0; //error flag

        index_peak_x = 0;
        index_peak_v = 0; //index of the peak of x or w to search right now.

        arr_v[0] = 0.0;
        arr_x[0] = 0.0;

        mean_v = 0.0;
        mean_x = 0.0;
        std_v = 0.0;
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
            next_x = x + dt * (a + b * (scale * pow(v, num_pow) + shift) - k_x * x);

            int adjusted_index = (i+1)%size;
            arr_v[adjusted_index] = next_v;
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
                    error_v = 0;
                    max_index_v = 0;
                    max_value_v = -INFINITY; //to avoid no peak(local maximum) being found

                    for (j = 0; j <= 20; j++){
                        if (arr_v[index%size] < arr_v[(index-10+j)%size]){
                            error_v = 1;
                        }
                    }

                    if (error_v == 0){
                        max_value_v = arr_v[index%size];
                        max_index_v = index;
                    }
                }else {
                    error_v = 0;
                        
                    if (arr_v[index%size] > max_value_v) {
                            
                        for (j = 0; j <= 20; j++){
                            if (arr_v[index%size] < arr_v[(index-10+j+size)%size]){
                                error_v = 1;
                            }
                        }

                        if (error_v == 0){
                            max_value_v = arr_v[index%size];
                            max_index_v = index;
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

        ids_peak_v[index_peak_v] = max_index_v;
        ids_peak_x[index_peak_x] = max_index_x;

        previous_peak_v = max_index_v;
        previous_peak_x = max_index_x;

        //printf("previous_x%d\n", previous_peak_x);

        index_v = ids_peak_v[0];
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
            next_x = x + dt * (a + b * (scale * pow(v, num_pow) + shift) - k_x * x);

            //printf("diff=%lf\n", next_w-arr_x[i%size]);

            int adjusted_index = (i+1)%size;
            arr_v[adjusted_index] = next_v;
            arr_x[adjusted_index] = next_x;

            u = next_u;
            v = next_v;
            w = next_w;
            x = next_x;

            //search peaks of w
            if ((previous_peak_v+min_period <= index_v) && (index_v <= previous_peak_v+max_period)) {
                if ((previous_peak_v+min_period) == index_v){
                    error_v = 0;
                    max_index_v = 0;
                    max_value_v = -INFINITY;//to avoid no peak(local maximum) being found

                    for (int j = 0; j <= 20; j++){
                        int adjusted_index_v = (index_v - 10 + j + size) % size;
                        if (arr_v[index_v % size] < arr_v[adjusted_index_v]) {
                            error_v = 1;
                        }
                    }
                    if (error_v == 0){
                        max_value_v = arr_v[index_v%size];
                        max_index_v = index_v;
                    }
                } else {
                    error_v = 0;
                    if (arr_v[index_v%size] > max_value_v) {
                        for (int j = 0; j <= 20; j++){
                            int adjusted_index_v = (index_v - 10 + j + size) % size;
                            if (arr_v[index_v % size] < arr_v[adjusted_index_v]) {
                                error_v = 1;
                            }
                        }
                        if (error_v == 0){
                            max_value_v = arr_v[index_v%size];
                            max_index_v = index_v;
                        }
                    }
                }

                //check if for loop reaches the end of a range to search peaks
                if (index_v == previous_peak_v+max_period) {
                    if (max_index_v == 0){//if no peak are found in the range of x, the search range will be expanded. max_idex_w still being 0 means no peaks is found.
                        previous_peak_v = previous_peak_v + (max_period - min_period);
                    } else {
                        index_peak_v++;
                        ids_peak_v[index_peak_v] =  max_index_v;
                        previous_peak_v = max_index_v;
                    }
                }
            }
            index_v++;
        
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
            if ((count <= index_peak_v) && (count <= index_peak_x)){
                //printf("hotateeeeeeeeeeeeeeeeee\n");
                break;
            }
        }

        measure_period(ids_peak_v, count, sample_period, dt, period_v, distance);
        measure_period(ids_peak_x, count, sample_period, dt, period_x, distance);

        mean_v = 0.0; 
        mean_x = 0.0;
        std_v = 0.0;
        std_x = 0.0;

        for (i = 0; i < sample_period; i++) {
            mean_v += period_v[i];
            mean_x += period_x[i];
        }
        
        mean_v /= sample_period;
        mean_x /= sample_period;
        for (i = 0; i < sample_period; i++) {
            std_v += pow(period_v[i] - mean_v, 2);
            std_x += pow(period_x[i] - mean_x, 2);
        }

        std_v = sqrt(std_v / sample_period);
        std_x = sqrt(std_x / sample_period);

        cv_v = cv_v + std_v / mean_v;
        cv_x = cv_x + std_x / mean_x;

    }

    *CV_v = cv_v/repetetion;
    *CV_x = cv_x/repetetion;
}

void CV_w(double a, double b, double scale, double shift, double num_pow,  double T, double k_u, double k_v, double k_w, double k_x, int step, double dt, double e, double D, int count, int start_time, int distance, int sample_period, int repetetion, int base_seed, double *CV_w, double *CV_x) {
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
                        /*
                        if (arr_w[index_w%size] < arr_w[(index_w-10+j+size)%size]){
                            error_w = 1;
                        }
                        */
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
                            /*
                            if (arr_w[index_w%size] < arr_w[(index_w-10+j+size)%size]){
                                error_w = 1;
                            }
                            */
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
                        //printf("ids_peak_w[i + 1]=%d\n", ids_peak_w[index_peak_w]);
                    }
                }
            }
            index_w++;
        
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
            if ((count <= index_peak_w) && (count <= index_peak_x)){
                //printf("hotateeeeeeeeeeeeeeeeee\n");
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
            //printf("period=%lf\n", diff);
            count++;
        }

        if (count >= sample_period) {
            //printf("count_getout=%d\n", count);
            break;
        }
    }
}

void writedata_2d(char *file, char **header, double **data, int rows, int columns) {
    FILE *fp = fopen(file, "w");

    if (fp == NULL) {
        perror("con't read a file");
        return;
    }

    //write header
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