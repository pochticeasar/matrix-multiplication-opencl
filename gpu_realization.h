#pragma once

int calculate1(cl_device_id device_id, cl_float* a, cl_float* b, cl_float* c, size_t n, size_t m, size_t k);

int calculate2(cl_device_id device_id, const cl_float* a, const cl_float* b, cl_float* c, size_t n, size_t m, size_t k);

int calculate3(cl_device_id device_id, const cl_float* a, const cl_float* b, cl_float* c, size_t n, size_t m, size_t k);
