#include <immintrin.h>

const char* dgemm_desc = "Simple blocked vectorized dgemm, mm512.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 40
#define BLOCK_AVX 5
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */

// Rearrange loop order and using constant(relative to inner loop) to replace variables
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    int P1,P2,Q1,Q2;
    __m256d tempB;
    __m512d vectorA[BLOCK_SIZE * BLOCK_AVX],  vectorC[BLOCK_SIZE * BLOCK_AVX];
	__m512d AA,CC,BB;
	int MM = M / 8;
	
	for (int j = 0; j < K; ++j) {
		P1 = j * MM;
		P2 = j * lda;
		for (int i = 0; i < M; i += 8)
			vectorA[P1 + i / 8] = _mm512_loadu_pd(A + i + P2);
	}
		
		
	for (int j = 0; j < N; ++j) {
		P1 = j * MM;
		P2 = j * lda;
		for (int i = 0; i < M; i += 8)
			vectorC[P1 + i / 8] = _mm512_loadu_pd(C + i + P2);
	}
		
    for (int j = 0; j < N; ++j) {
		Q1 = j * MM;
        Q2 = j * lda;
        for (int k = 0; k < K; ++k) {
            P1 = k * MM;
            // mm512 does not have load + broadcast function, so this has to be done in two steps
            tempB = _mm256_broadcast_sd(B + (k + Q2));
            BB = _mm512_broadcast_f64x4(tempB);
            for (int i = 0; i < MM; ++i) {
                // C[i + Q] += A[i + P] * vectorB;
                AA = vectorA[P1 + i];
                CC = vectorC[Q1 + i];
                // A * B
                AA = _mm512_mul_pd(AA, BB);
                // C + (A * B)
                vectorC[Q1 + i] = _mm512_add_pd(CC, AA);
                
            }
        }
    }
	for (int j = 0; j < N; ++j) {
		Q1 = j * MM;
		Q2 = j * lda;
		for (int i = 0; i < M; i += 8)
			// store C
            _mm512_storeu_pd(C + (i + Q2), vectorC[Q1 + i / 8]);
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
