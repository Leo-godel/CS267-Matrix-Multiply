#include <immintrin.h>
#include <xmmintrin.h>
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
static inline __attribute__((optimize("unroll-loops"))) void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    int P1,P2,Q1,Q2;
    __m512d vectorA[BLOCK_SIZE * BLOCK_AVX],  vectorC[BLOCK_SIZE * BLOCK_AVX];
	__m512d AA,CC,BB;
	int MM = M / 8, MMD = MM * 8;
	double bb;
	double cutA[BLOCK_SIZE][8], cutC[BLOCK_SIZE][8];

	for (int j = 0; j < K; ++j) {
		P1 = j * MM;
		P2 = j * lda;
		for (int i = 0; i < MMD; i += 8)
			vectorA[P1 + i / 8] = _mm512_loadu_pd(A + i + P2);
		for (int i = MMD; i < M; i++)
			cutA[j][i - MMD] = A[i + P2];
	}
		
		
	for (int j = 0; j < N; ++j) {
		P1 = j * MM;
		P2 = j * lda;
		for (int i = 0; i < MMD; i += 8)
			vectorC[P1 + i / 8] = _mm512_loadu_pd(C + i + P2);
		for (int i = MMD; i < M; i++)
			cutC[j][i - MMD] = C[i + P2];
	}
		
	
	if (MM == 5)
	{
		for (int j = 0; j < N; ++j) {
			Q1 = j * MM;
			Q2 = j * lda;
			for (int k = 0; k < K; ++k) {
				P1 = k * MM;
				// mm512 does not have load + broadcast function, so this has to be done in two steps
				BB = _mm512_set1_pd(B[k + Q2]);
				
				// unroll the loop
					// C[i + Q] += A[i + P] * vectorB;
					AA = vectorA[P1];
					CC = vectorC[Q1];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1] = _mm512_add_pd(CC, AA);
					
					AA = vectorA[P1 + 1];
					CC = vectorC[Q1 + 1];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1 + 1] = _mm512_add_pd(CC, AA);
					
					AA = vectorA[P1 + 2];
					CC = vectorC[Q1 + 2];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1 + 2] = _mm512_add_pd(CC, AA);
					
					AA = vectorA[P1 + 3];
					CC = vectorC[Q1 + 3];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1 + 3] = _mm512_add_pd(CC, AA);
					
					AA = vectorA[P1 + 4];
					CC = vectorC[Q1 + 4];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1 + 4] = _mm512_add_pd(CC, AA);
				
			}
		}
	}
	else
	{
		for (int j = 0; j < N; ++j) {
			Q1 = j * MM;
			Q2 = j * lda;
			for (int k = 0; k < K; ++k) {
				P1 = k * MM;
				bb = B[k + Q2];
				BB = _mm512_set1_pd(bb);
				for (int i = 0; i < MM; ++i) {
					// C[i + Q] += A[i + P] * vectorB;
					AA = vectorA[P1 + i];
					CC = vectorC[Q1 + i];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1 + i] = _mm512_add_pd(CC, AA);
                
				}
				for (int i = MMD; i < M; ++i) 
					cutC[j][i - MMD] += cutA[k][i - MMD] * bb;
			}
		}
	}
	
	for (int j = 0; j < N; ++j) {
		Q1 = j * MM;
		Q2 = j * lda;
		for (int i = 0; i < MMD; i += 8)
			// store C
            _mm512_storeu_pd(C + (i + Q2), vectorC[Q1 + i / 8]);
	}

	if (MMD != M)
		for (int j = 0; j < N; ++j) {
			Q2 = j * lda;
			for (int i = MMD; i < M; ++i)
				C[Q2 + i] = cutC[j][i - MMD];
		}
}

static inline __attribute__((optimize("unroll-loops"))) void do_block2(int lda, int M, int N, int K, double* A, double* B, double* C) {
    int P1,P2,Q1,Q2;
 
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
		
	
	if (MM == 5)
	{
		for (int j = 0; j < N; ++j) {
			Q1 = j * MM;
			Q2 = j * lda;
			for (int k = 0; k < K; ++k) {
				P1 = k * MM;
				// mm512 does not have load + broadcast function, so this has to be done in two steps
				BB = _mm512_set1_pd(B[k + Q2]);
				
				// unroll the loop
					// C[i + Q] += A[i + P] * vectorB;
					AA = vectorA[P1];
					CC = vectorC[Q1];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1] = _mm512_add_pd(CC, AA);
					
					AA = vectorA[P1 + 1];
					CC = vectorC[Q1 + 1];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1 + 1] = _mm512_add_pd(CC, AA);
					
					AA = vectorA[P1 + 2];
					CC = vectorC[Q1 + 2];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1 + 2] = _mm512_add_pd(CC, AA);
					
					AA = vectorA[P1 + 3];
					CC = vectorC[Q1 + 3];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1 + 3] = _mm512_add_pd(CC, AA);
					
					AA = vectorA[P1 + 4];
					CC = vectorC[Q1 + 4];
					// A * B
					AA = _mm512_mul_pd(AA, BB);
					// C + (A * B)
					vectorC[Q1 + 4] = _mm512_add_pd(CC, AA);
				
			}
		}
	}
	else
	{
		for (int j = 0; j < N; ++j) {
			Q1 = j * MM;
			Q2 = j * lda;
			for (int k = 0; k < K; ++k) {
				P1 = k * MM;
				BB = _mm512_set1_pd(B[k + Q2]);
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
__attribute__((optimize("unroll-loops"))) void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int j = 0; j < lda; j += BLOCK_SIZE) {
        // For each block-column of B
        for (int k = 0; k < lda; k += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int i = 0; i < lda; i += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                if (M % 8 != 0) do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
		else do_block2(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}
