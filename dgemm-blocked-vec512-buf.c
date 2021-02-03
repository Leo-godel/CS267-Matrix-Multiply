#include <immintrin.h>

const char* dgemm_desc = "Simple blocked vectorized dgemm, mm512.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#define BLOCK_AVX 4
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */

// Rearrange loop order and using constant(relative to inner loop) to replace variables
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    int P, QB, QC, i, j, k;
    __m256d tempB;
    __m512d vectorA, vectorB, vectorC;
	double* A_buf __attribute__((aligned(32))) = malloc(M * K * sizeof(double));
    double* B_buf __attribute__((aligned(32))) = malloc(N * K * sizeof(double));
    double* C_buf __attribute__((aligned(32))) = malloc(M * N * sizeof(double));
    
    // copy over the slice of A
    for (k = 0; k < K; ++k) {
        for (i = 0; i < M; ++i) {
            A_buf[i + k * M] = A[i + k * lda];
        }
    }
    // copy over the slice of B
    for (j = 0; j < N; ++j) {
        for (k = 0; k < K; ++k) {
            B_buf[k + j * N] = B[k + j * lda];
        }
    }
    // copy over the slice of C
    for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
            C_buf[i + j * M] = C[i + j * lda];
        }
    }
		
    for (j = 0; j < N; ++j) {
        QC = j * M;
        QB = j * N;
        for (k = 0; k < K; ++k) {
            P = k * M;
            // mm512 does not have load + broadcast function, so this has to be done in two steps
            tempB = _mm256_broadcast_sd(B_buf + (k + QB));
            vectorB = _mm512_broadcast_f64x4(tempB);
            for (i = 0; i < M/8 * 8; i += 8) {
                // C[i + Q] += A[i + P] * vectorB;
                vectorA = _mm512_loadu_pd(A_buf + (i + P));
                vectorC = _mm512_loadu_pd(C_buf + (i + QC));
                // A * B
                vectorA = _mm512_mul_pd(vectorA, vectorB);
                // C + (A * B)
                vectorC = _mm512_add_pd(vectorC, vectorA);
                // store results in C buffer
                _mm512_storeu_pd(C_buf + (i + QC), vectorC);
            }
        }
    }

    // copy back results from C buffer into C
	for (j = 0; j < N; ++j) {
        for (i = 0; i < M; ++i) {
            C[i + j * lda] = C_buf[i + j * M];
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}
