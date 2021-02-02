void square_dgemm(int n, double* A, double* B, double* C) {
    int P,Q;
    double R;
    for (int j = 0; j < n; ++j) {
        Q = j * n;
        for (int k = 0; k < n; ++k) {
            R = B[k + Q];
            P = k * n;
            for (int i = 0; i < n; ++i)
                C[i + Q] += A[i + P] * R;
        }
    }


}