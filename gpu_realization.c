//
// Created by Файзиева Юлия on 08.09.2024.
//
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>
#include <stdio.h>

#endif
#include "gpu_realization.h"


#define CONCAT3(a, b, c)  "-DTS2=" #a " -DTS3=" #b " -DWI=" #c
#define CONCAT(a, b, c) CONCAT3(a, b, c)

#ifndef LOCAL2
#define LOCAL2 16
#endif

#ifndef LOCAL3
#define LOCAL3 32
#endif

#ifndef ITEM
#define ITEM 8
#endif

cl_uint round_to(cl_uint n, cl_uint m) {
    cl_uint del = n/m + (n % m == 0 ? 0 : 1);
    return del * m;
}

int calculate1(cl_device_id device_id, cl_float* a, cl_float* b, cl_float* c, size_t n, size_t m, size_t k) {
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    FILE *file = fopen("mmul.cl", "rb");
    if (file == NULL) {
        return 1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 1;
    }
    size_t size = ftell(file);
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 1;
    }

    char *source = malloc(size + 1);
    if (source == NULL) {
        fclose(file);
        return 1;
    }
    if (fread(source, 1, size, file) != size) {
        fclose(file);
        free(source);
        return 1;
    }
    if (fclose(file) == EOF) {
        free(source);
        return 1;
    }
    source[size] = '\0';
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, (const size_t *)&size, NULL);
    cl_int build_err = clBuildProgram(program, 1, &device_id,  CONCAT(LOCAL2,LOCAL3, ITEM), NULL, NULL);
    size_t log_size;
    if (build_err != 0) {
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        clReleaseProgram(program);
        fprintf(stderr, "%s\n", log);
        free(log);
        return 1;
    }
    cl_int error = 0;
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    if (error != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "matrix_multiplication1", &error);
    if (error != CL_SUCCESS) {
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(cl_float) * m * k, NULL, &error);
    if (error != CL_SUCCESS) {
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(cl_float) * k * n, NULL, &error);
    if (error != CL_SUCCESS) {
        clReleaseMemObject(a_mem);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  sizeof(cl_float) * m * n, NULL, &error);
    if (error != CL_SUCCESS) {
        clReleaseMemObject(a_mem);
        clReleaseMemObject(b_mem);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }


    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &n);
    clSetKernelArg(kernel, 4, sizeof(cl_uint), &k);
    clSetKernelArg(kernel, 5, sizeof(cl_uint), &m);
    cl_event events[4];
    clEnqueueWriteBuffer(command_queue, a_mem, CL_FALSE, 0,
                         sizeof(cl_float) *  m * k, a, 0, NULL, &events[0]);
    clEnqueueWriteBuffer(command_queue, b_mem, CL_FALSE, 0,
                         sizeof(cl_float) * k * n, b, 0, NULL, &events[1]);
    size_t global_work_size[] = {n, m};
    error = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL,
                           0, NULL, &events[2]);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel is not succesful: %d\n", error);
        clReleaseMemObject(a_mem);
        clReleaseMemObject(b_mem);
        clReleaseMemObject(c_mem);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        free(source);
        return 1;

    }
    clEnqueueReadBuffer(command_queue, c_mem, CL_TRUE, 0,
                        sizeof(cl_float) * m * n, c, 0, NULL, &events[3]);

    clFinish(command_queue);

    cl_ulong time_start, time_end;
    cl_double time_with_memory = 0;
    cl_double time_without_memory = 0;
    for (int i = 0; i < 4; i++) {
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(time_start),
                                &time_start, NULL);
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(time_end),
                                &time_end, NULL);
        time_with_memory += (double)(time_end - time_start) / 1000000.0;
        if (i == 2) {
            time_without_memory = (double)(time_end - time_start) / 1000000.0;
        }
    }
    printf("Time: %g\t%g\n", time_without_memory, time_with_memory);
    clReleaseMemObject(a_mem);
    clReleaseMemObject(b_mem);
    clReleaseMemObject(c_mem);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}

int calculate2(cl_device_id device_id, const cl_float* a, const cl_float* b, cl_float* c, size_t n, size_t m, size_t k) {
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    FILE *file = fopen("mmul.cl", "rb");
    if (file == NULL) {
        return 1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 1;
    }
    size_t size = ftell(file);
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 1;
    }

    char *source = malloc(size + 1);
    if (source == NULL) {
        fclose(file);
        return 1;
    }
    if (fread(source, 1, size, file) != size) {
        fclose(file);
        free(source);
        return 1;
    }
    if (fclose(file) == EOF) {
        free(source);
        return 1;
    }
    source[size] = '\0';
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, (const size_t *)&size, NULL);
    cl_int build_err = clBuildProgram(program, 1, &device_id, CONCAT(LOCAL2, LOCAL3, ITEM), NULL, NULL);
    size_t log_size;
    if (build_err != 0) {
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        clReleaseProgram(program);
        fprintf(stderr, "%s\n", log);
        free(log);
        return 1;
    }
    cl_int error = 0;
    cl_uint m_round = round_to(m, LOCAL2);
    cl_uint n_round = round_to(n, LOCAL2);
    cl_uint k_round = round_to(k, LOCAL2);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    if (error != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "matrix_multiplication2", &error);
    if (error != CL_SUCCESS) {
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_float* a_round = calloc(m_round * k_round, sizeof(cl_float));
    if (a_round == NULL) {
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    for (cl_int i = 0; i < m; i++) {
        for (cl_int j = 0; j < k; j++) {
            a_round[i * k_round + j] = a[i * k + j];
        }
    }
    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(cl_float) * m_round * k_round, NULL, &error);
    if (error != CL_SUCCESS) {
        free(a_round);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_float* b_round = calloc(n_round * k_round, sizeof(cl_float));
    if (b_round == NULL) {
        free(a_round);
        clReleaseMemObject(a_mem);

        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    for (cl_int i = 0; i < k; i++) {
        for (cl_int j = 0; j < n; j++) {
            b_round[i * n_round + j] = b[i * n + j];
        }
    }
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(cl_float) * k_round * n_round, NULL, &error);
    if (error != CL_SUCCESS) {
        free(a_round);
        free(b_round);
        clReleaseMemObject(a_mem);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_float* c_round = calloc(n_round * m_round, sizeof(cl_float));
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  sizeof(cl_float) * m_round * n_round, NULL, &error);
    if (error != CL_SUCCESS) {
        clReleaseMemObject(a_mem);
        clReleaseMemObject(b_mem);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        clReleaseProgram(program);
        free(source);
        return 1;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_round);
    clSetKernelArg(kernel, 4, sizeof(cl_uint), &k_round);
    clSetKernelArg(kernel, 5, sizeof(cl_uint), &m_round);
    cl_event events[4];
    clEnqueueWriteBuffer(command_queue, a_mem, CL_FALSE, 0,
                         sizeof(cl_float) * m_round * k_round, a_round, 0, NULL, &events[0]);
    clEnqueueWriteBuffer(command_queue, b_mem, CL_FALSE, 0,
                         sizeof(cl_float) * k_round * n_round, b_round, 0, NULL, &events[1]);
    size_t global_work_size[] = {n_round, m_round};
    size_t local_work_size[] = {LOCAL2, LOCAL2};

    error = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size,
                                   0, NULL, &events[2]);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel is not succesful: %d\n", error);
        clReleaseMemObject(a_mem);
        clReleaseMemObject(b_mem);
        clReleaseMemObject(c_mem);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    clEnqueueReadBuffer(command_queue, c_mem, CL_TRUE, 0,
                        sizeof(cl_float) * m_round * n_round, c_round, 0, NULL, &events[3]);

    clFinish(command_queue);
    for (cl_int i = 0; i < m; i++) {
        for (cl_int j = 0; j < n; j++) {
            c[i * n + j] = c_round[i * n_round + j];
        }
    }

    cl_ulong time_start, time_end;
    cl_double time_with_memory = 0;
    cl_double time_without_memory = 0;
    for (int i = 0; i < 4; i++) {
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(time_start),
                                &time_start, NULL);
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(time_end),
                                &time_end, NULL);
        time_with_memory += (double)(time_end - time_start) / 1000000.0;
        if (i == 2) {
            time_without_memory = (double)(time_end - time_start) / 1000000.0;
        }
    }
    printf("Time: %g\t%g\n", time_without_memory, time_with_memory);
    printf("LOCAL_WORK_SIZE [%zu, %zu]\n", local_work_size[0], local_work_size[1]);
    clReleaseMemObject(a_mem);
    clReleaseMemObject(b_mem);
    clReleaseMemObject(c_mem);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}

int calculate3(cl_device_id device_id, const cl_float* a, const cl_float* b, cl_float* c, size_t n, size_t m, size_t k) {
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);
    FILE *file = fopen("mmul.cl", "rb");
    if (file == NULL) {
        return 1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return 1;
    }
    size_t size = ftell(file);
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 1;
    }

    char *source = malloc(size + 1);
    if (source == NULL) {
        fclose(file);
        return 1;
    }
    if (fread(source, 1, size, file) != size) {
        fclose(file);
        free(source);
        return 1;
    }
    if (fclose(file) == EOF) {
        free(source);
        return 1;
    }
    source[size] = '\0';
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, (const size_t *)&size, NULL);
    cl_int build_err = clBuildProgram(program, 1, &device_id, CONCAT(LOCAL2,LOCAL3, ITEM), NULL, NULL);
    size_t log_size;
    if (build_err != 0) {
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        clReleaseProgram(program);
        fprintf(stderr, "%s\n", log);
        free(log);
        return 1;
    }
    cl_int error = 0;
    cl_uint m_round = round_to(m, LOCAL3);
    cl_uint n_round = round_to(n, LOCAL3);
    cl_uint k_round = round_to(k, LOCAL3);
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    if (error != CL_SUCCESS) {
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_kernel kernel = clCreateKernel(program, "matrix_multiplication3", &error);
    if (error != CL_SUCCESS) {
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_float* a_round = calloc(m_round * k_round, sizeof(cl_float));
    if (a_round == NULL) {
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    for (cl_int i = 0; i < m; i++) {
        for (cl_int j = 0; j < k; j++) {
            a_round[i * k_round + j] = a[i * k + j];
        }
    }
    cl_mem a_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(cl_float) * m_round * k_round, NULL, &error);
    if (error != CL_SUCCESS) {
        free(a_round);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_float* b_round = calloc(n_round * k_round, sizeof(cl_float));
    if (b_round == NULL) {
        free(a_round);
        clReleaseMemObject(a_mem);

        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    for (cl_int i = 0; i < k; i++) {
        for (cl_int j = 0; j < n; j++) {
            b_round[i * n_round + j] = b[i * n + j];
        }
    }
    cl_mem b_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  sizeof(cl_float) * k_round * n_round, NULL, &error);
    if (error != CL_SUCCESS) {
        free(a_round);
        free(b_round);
        clReleaseMemObject(a_mem);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }
    cl_float* c_round = calloc(n_round * m_round, sizeof(cl_float));
    cl_mem c_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                  sizeof(cl_float) * m_round * n_round, NULL, &error);
    if (error != CL_SUCCESS) {
        clReleaseMemObject(a_mem);
        clReleaseMemObject(b_mem);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;
    }


    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem);
    clSetKernelArg(kernel, 3, sizeof(cl_uint), &n_round);
    clSetKernelArg(kernel, 4, sizeof(cl_uint), &k_round);
    clSetKernelArg(kernel, 5, sizeof(cl_uint), &m_round);
    cl_event events[4];
    clEnqueueWriteBuffer(command_queue, a_mem, CL_FALSE, 0,
                         sizeof(cl_float) * m_round * k_round, a_round, 0, NULL, &events[0]);
    clEnqueueWriteBuffer(command_queue, b_mem, CL_FALSE, 0,
                         sizeof(cl_float) * k_round * n_round, b_round, 0, NULL, &events[1]);
    size_t global_work_size[] = {n_round/ITEM, m_round};
    size_t local_work_size[] = {LOCAL3/ITEM, LOCAL3};

    error = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, local_work_size,
                                   0, NULL, &events[2]);
    if (error != CL_SUCCESS) {
        fprintf(stderr, "clEnqueueNDRangeKernel is not succesful: %d\n", error);
        clReleaseMemObject(a_mem);
        clReleaseMemObject(b_mem);
        clReleaseMemObject(c_mem);
        clReleaseKernel(kernel);
        clReleaseCommandQueue(command_queue);
        clReleaseProgram(program);
        clReleaseContext(context);
        free(source);
        return 1;

    }
    clEnqueueReadBuffer(command_queue, c_mem, CL_TRUE, 0,
                        sizeof(cl_float) * m_round * n_round, c_round, 0, NULL, &events[3]);

    clFinish(command_queue);
    for (cl_int i = 0; i < m; i++) {
        for (cl_int j = 0; j < n; j++) {
            c[i * n + j] = c_round[i * n_round + j];
        }
    }

    cl_ulong time_start, time_end;
    cl_double time_with_memory = 0;
    cl_double time_without_memory = 0;
    for (int i = 0; i < 4; i++) {
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(time_start),
                                &time_start, NULL);
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(time_end),
                                &time_end, NULL);
        time_with_memory += (double)(time_end - time_start) / 1000000.0;
        if (i == 2) {
            time_without_memory = (double)(time_end - time_start) / 1000000.0;
        }
    }
    printf("Time: %g\t%g\n", time_without_memory, time_with_memory);
    printf("LOCAL_WORK_SIZE [%zu, %zu]\n", local_work_size[0], local_work_size[1]);
    printf("WI_WORK %i\n", ITEM);
    clReleaseMemObject(a_mem);
    clReleaseMemObject(b_mem);
    clReleaseMemObject(c_mem);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(command_queue);
    clReleaseProgram(program);
    clReleaseContext(context);
    return 0;
}
