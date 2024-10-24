/* まずSFMT.hをインクルード */
#include "SFMT.h"

int main(int argc, char *argv[]) {
    /* 状態を保持する構造体 */
    sfmt_t sfmt;

    /* シードを指定して初期化 */
    int seed = 0;
    sfmt_init_gen_rand(&sfmt, seed);

    origin_sfmt = sfmt

    //first random
    uint64_t v = sfmt_genrand_uint64(&origin_sfmt);
    double x = sfmt_to_res53(v);
    printf("%f\n",x);

    //second random
    v = sfmt_genrand_uint64(&origin_sfmt);
    x = sfmt_to_res53(v);
    printf("%f\n",x);

    return 0;
}
