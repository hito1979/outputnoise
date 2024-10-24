#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//**********************************************************************************
//グローバル変数
int N = 5;
int iteration = 2e+8;

//**********************************************************************************
//定義
char base_path[200] = "..";
int length = 10;
int num;

//**********************************************************************************************************************************************************************************

int main() {
    //宣言
    char file_sample[200], file_normalized[200];

    num = iteration / length;

    // header配列にメモリを動的に割り当て
    char **header;
    header = (char **)malloc((2 * N + 1) * sizeof(char *));
    for (int i = 0; i < 2 * N + 1; i++) {
        header[i] = (char *)malloc(20 * sizeof(char)); // 各要素に20文字分のメモリを割り当て
    }

    double **sample = (double**)malloc(sizeof(double*) * (length));
    for (int i = 0; i < length; ++i) {
        sample[i] = (double*)malloc(sizeof(double) * (2*N+1));
    }

    //headerの定義
    for (int i = 0; i < 2*N+1; i++) {

        if (i==0){
            strcpy(header[i], "CV");
        }

        else if (0 < i && i <= N){
            char temp[20]; // 数値を一時的に格納する文字列
            sprintf(temp, "%s%d", "A_", i); // 数値を文字列に変換
            strcpy(header[i], temp);
        }

        else {
            char temp[20]; // 数値を一時的に格納する文字列o
            sprintf(temp, "%s%d", "B_", i-N); // 数値を文字列に変換
            strcpy(header[i], temp);
        }

    }

    //##################################################################################
    //メイン

    for (int i = 6000000; i <= num; i++) {

        sprintf(file_normalized, "%s/Result/N=%d/Normalise/normalised_sample_%d.csv", base_path, N, i);

        printf("Deleting: %s\n", file_normalized);
        if (remove(file_normalized) == 0) {
            printf("Deleted successfully: %s\n", file_normalized);
        } else {
            printf("Failed to delete: %s\n", file_normalized);
        }
    }

    // メモリの解放
    for (int i = 0; i < 2 * N + 1; i++) {
        free(header[i]);
    }
    free(header);

    for (int i = 0; i < length; ++i) {
        free(sample[i]);
    }
    free(sample);

    return 0;

}
