#include "gemm.h"
#include "utils.h"
void pm(int M, int N, float* A)
{
    int i, j;
    printf("\n  ");
    for (i = 0; i < N; ++i)
        printf(" %d     |",i+1);
    printf("\n");
    for (i = 0; i < M; ++i) {
        printf("%d| ", i + 1);
        for (j = 0; j < N; ++j) {
            printf("%2.4f| ", A[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}
float* random_matrix(int rows, int cols)
{
    int i;
    float* m = (float*)xcalloc(rows * cols, sizeof(float));
    for (i = 0; i < rows * cols; ++i) {
        m[i] = (float)rand() / RAND_MAX;
    }
    return m;
}
void test_gemm(int m, int k, int n) {
    srand(0);
    float* a;
    a = random_matrix(m, k);
    int lda =  k;
    float* b;
    b = random_matrix(k, n);
    int ldb = n;

    float* c = random_matrix(m, n);
    float* c_gpu = random_matrix(m, n);
    memset(c, 0, m * n * sizeof(float));
    memset(c_gpu, 0, m * n * sizeof(float));
    int i;
    printf("\n\nmat a\n");
    pm(k, m, a);

    printf("\n\nmat b\n");
    pm(m, k, b);


    
    printf("GPU\n");
    gemm_gpu(0, 0, m, n, k, 1, a, lda, b, ldb, 1, c_gpu, n);
    //gemm_gpu(0, 0, m, n, k, 1, a, lda, b, ldb, 1, c_gpu, n);
    pm(m, n, c_gpu);
    //* 矩阵计算，完成C = ALPHA * A * B + BETA * C 矩阵计算，最后的输出为C
    gemm_cpu(0, 0, m, n, k, 1, a, lda, b, ldb, 1, c, n);
    //gemm_cpu(0, 0, m, n, k, 1, a, lda, b, ldb, 1, c, n);
    printf("\n\nCPU\n");
    
    pm(m, n, c);
    double sse = 0;
    for (i = 0; i < m * n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i] - c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n", m, k, k, n, 0, 0, sse / (m * n));
    //printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d\n", m, k, k, n, 0, 0);
    free(a);
    free(b);
    free(c);
    free(c_gpu);
}

int main(int argc,char** argv) {
    int rows, cols,m,k,n;
    rows = atoi(argv[1]); cols = atoi(argv[2]);
    m = atoi(argv[1]); k = atoi(argv[2]); n = atoi(argv[3]);
    float* xx = random_matrix(rows, cols);
    int i;
    for (i = 0; i < (rows * cols); i++) {
        if (!(i % cols))printf(" \n");
        printf("%f ", xx[i]);
    }
    printf(" \n ");
    test_gemm(m, k, n);
    //test_gpu_accuracy(0,0,m, k, n);


    is_avx();
    is_fma_avx2();
}

/*
0.840188 0.394383 0.783099 0.798440 0.911647  
0.197551 0.335223 0.768230 0.277775 0.553970  
0.477397 0.628871 0.364784 0.513401 0.952230  
0.916195 0.635712 0.717297 0.141603 0.606969  
 

mat a

   1     | 2     | 3     | 4     |
1| 0.8402| 0.3944| 0.7831| 0.7984| 
2| 0.9116| 0.1976| 0.3352| 0.7682| 
3| 0.2778| 0.5540| 0.4774| 0.6289| 
4| 0.3648| 0.5134| 0.9522| 0.9162| 
5| 0.6357| 0.7173| 0.1416| 0.6070| 



mat b

   1     | 2     | 3     | 4     | 5     |
1| 0.0163| 0.2429| 0.1372| 0.8042| 0.1567| 
2| 0.4009| 0.1298| 0.1088| 0.9989| 0.2183| 
3| 0.5129| 0.8391| 0.6126| 0.2960| 0.6376| 
4| 0.5243| 0.4936| 0.9728| 0.2925| 0.7714| 

GPU

   1     | 2     | 3     | 4     | 5     | 6     |
1| 1.0365| 1.4160| 2.1655| 2.6249| 1.1036| 3.0069| 
2| 0.7556| 0.7214| 1.4455| 1.3578| 0.7319| 1.8814| 
3| 0.7328| 1.0240| 1.9658| 1.9828| 0.8493| 2.4356| 
4| 0.7503| 0.8272| 1.7829| 1.9184| 0.9227| 2.3010| 

 Used AVX 
 Used FMA & AVX2 


CPU

   1     | 2     | 3     | 4     | 5     | 6     |
1| 1.0365| 1.4160| 2.1655| 2.6249| 1.1036| 3.0069| 
2| 0.7556| 0.7214| 1.4455| 1.3578| 0.7319| 1.8814| 
3| 0.7328| 1.0240| 1.9658| 1.9828| 0.8493| 2.4356| 
4| 0.7503| 0.8272| 1.7829| 1.9184| 0.9227| 2.3010| 

Matrix Multiplication 4x5 * 5x6, TA=0, TB=0: 5.03301e-15 SSE
*/
