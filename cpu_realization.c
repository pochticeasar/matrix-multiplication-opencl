#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef _WIN32
#include <Windows.h>
#else
#include <time.h>
#endif
#include "stdio.h"
#include "cpu_realization.h"

void calculate0(cl_float* a, cl_float* b, cl_float* c, size_t n, size_t m, size_t k) {

#ifdef _WIN32
    LARGE_INTEGER start, end;

    QueryPerformanceCounter(&start);

#else
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

#endif

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            c[i * n + j] = 0;
            for (size_t t = 0; t < k; t++) {
                c[i * n + j] += a[i * k + t] * b[t * n + j]; //c[i][j] += a[t][i] * b[t][i]
            }
        }
    }

#ifdef _WIN32
    QueryPerformanceCounter(&end);
    LARGE_INTEGER Frequency, elapsed;

    QueryPerformanceFrequency(&Frequency);
    elapsed.QuadPart = end.QuadPart - start.QuadPart;

    elapsed.QuadPart *= 1000;
    // elapsed.QuadPart /= Frequency.QuadPart;

    printf("Time: %g\n", (float) elapsed.QuadPart/ (float ) Frequency.QuadPart);

#else
    clock_gettime(CLOCK_MONOTONIC, &end);
    printf("Time: %g\n", (((float)end.tv_sec * 1000) + ((float)end.tv_nsec / 1000000)) -
        (((float)start.tv_sec * 1000) + ((float)start.tv_nsec / 1000000)));
#endif
}