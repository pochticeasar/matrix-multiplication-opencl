#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else

#include <CL/cl.h>

#endif

#include "cpu_realization.h"
#include "gpu_realization.h"

#include <stdio.h>
#include <string.h>
#include <limits.h>

#define N 2
#define M 3
#define K 1
enum type_of_device {
    DGPU = 0,
    IGPU = 1,
    GPU = 2,
    CPU = 3,
    ALL = 4,
};
struct arguments {
    char *input_file_name;
    char *output_file_name;
    enum type_of_device device_type;
    int index_of_device;
    int number_of_realisation;
};

int print_output(struct arguments *arguments, size_t n, size_t m, const cl_float *c);

int get_int_by_enum_device(cl_device_id device_id) {
    cl_device_type device_type;
    clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
    switch (device_type) {
        case (CL_DEVICE_TYPE_GPU): {
            cl_bool unified_memory;
            clGetDeviceInfo(device_id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), &unified_memory, NULL);
            if (unified_memory) {
                return 1;
            } else {
                return 0;
            }
        }
        case (CL_DEVICE_TYPE_CPU): {
            return 3;
        }
        default: {
            return 4;
        }
    }
}

int compare_devices(const void *p1, const void *p2) {
    cl_device_id first = *(cl_device_id *) p1;
    cl_device_id second = *(cl_device_id *) p2;
    return get_int_by_enum_device(first) - get_int_by_enum_device(second);
}


cl_device_id get_platforms(struct arguments *arguments) {
    cl_uint number_of_platforms;
    if (clGetPlatformIDs(0, NULL, &number_of_platforms) != CL_SUCCESS) {
        fprintf(stderr, "Error while getting platforms\n");
        return NULL;
    }
    cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * number_of_platforms);
    if (platforms == NULL) {
        fprintf(stderr, "No platforms found\n");
        return NULL;
    }
    if (clGetPlatformIDs(number_of_platforms, platforms, NULL) != CL_SUCCESS) {
        free(platforms);
        fprintf(stderr, "Error while getting platforms\n");
        return NULL;
    }
    cl_uint number_of_devices = 0;
    for (size_t i = 0; i < number_of_platforms; i++) {
        cl_uint number_of_devices_by_platform = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &number_of_devices_by_platform) != CL_SUCCESS) {
            free(platforms);
            fprintf(stderr, "Error while getting devices\n");
            return NULL;
        }
        number_of_devices += number_of_devices_by_platform;
    }

    cl_device_id *devices = malloc(sizeof(cl_device_id) * number_of_devices);
    if (devices == NULL) {
        free(platforms);
        fprintf(stderr, "Error while getting devices\n");
        return NULL;
    }
    cl_uint number_of_devices_written = 0;
    for (size_t i = 0; i < number_of_platforms; i++) {
        cl_uint number_of_devices_by_platform = 0;
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &number_of_devices_by_platform) != CL_SUCCESS) {
            free(platforms);
            free(devices);
            fprintf(stderr, "Error while getting devices\n");
            return NULL;
        }

        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, number_of_devices_by_platform,
                           devices + number_of_devices_written, NULL) != CL_SUCCESS) {
            free(platforms);
            free(devices);
            fprintf(stderr, "Error while getting devices\n");
            return NULL;
        }
        number_of_devices_written += number_of_devices_by_platform;
    }
    qsort(&devices[0], number_of_devices, sizeof(cl_device_id), compare_devices);
    cl_device_id first_found = NULL;
    if (arguments->device_type != ALL) {
        for (cl_uint i = 0; i < number_of_devices; i++) {
            enum type_of_device device_type = get_int_by_enum_device(devices[i]);
            if ((arguments->device_type == GPU && (device_type == IGPU || device_type == DGPU))
                || device_type == arguments->device_type) {
                if (arguments->index_of_device == 0) {
                    return devices[i];
                } else if (first_found == NULL) {
                    first_found = devices[i];
                }
                arguments->index_of_device--;
            }
        }
        return first_found;
    } else {
        if (arguments->index_of_device >= number_of_devices) {
            return devices[0];
        } else {
            return devices[arguments->index_of_device];
        }
    }
}

int main(int argc, char **argv) {
    struct arguments arguments;
    arguments.output_file_name = NULL;
    arguments.input_file_name = NULL;
    arguments.number_of_realisation = -1;
    arguments.index_of_device = 0;
    arguments.device_type = ALL;
    if (argc < 2 || argc > 11) {
        fprintf(stderr, "Incorrect number of arguments\n");
        return 1;
    } else {
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--help")) {
                printf("%s", "lab1.exe < --input file_name > \\\n"
                             "         < --output file_name > \\\n"
                             "         [ --device-type { dgpu | igpu | gpu | cpu | all } ]\n"
                             "         [ --device-index index ]\n");
                return 0;
            } else if (!strcmp(argv[i], "--input")) {
                arguments.input_file_name = argv[++i];
            } else if (!strcmp(argv[i], "--output")) {
                arguments.output_file_name = argv[++i];
            } else if (!strcmp(argv[i], "--device-type")) {
                char *t = argv[++i];
                if (!strcmp(t, "dgpu")) {
                    arguments.device_type = DGPU;
                } else if (!strcmp(t, "igpu")) {
                    arguments.device_type = IGPU;
                } else if (!strcmp(t, "cpu")) {
                    arguments.device_type = CPU;
                } else if (!strcmp(t, "gpu")) {
                    arguments.device_type = GPU;
                } else if (!strcmp(t, "all")) {
                    arguments.device_type = ALL;
                } else {
                    fprintf(stderr, "Incorrect argument\n");
                    return 1;
                }
            } else if (!strcmp(argv[i], "--device-index")) {
                char *p;
                long conv = strtol(argv[++i], &p, 10);
                if (*p != '\0' || conv > INT_MAX || conv < 0) {
                    fprintf(stderr, "Incorrect argument\n");
                    return 1;
                }
                arguments.index_of_device = conv;
            } else if (!strcmp(argv[i], "--realization")) {
                char *p;
                long conv = strtol(argv[++i], &p, 10);
                if (*p != '\0' || conv > 3 || conv < 0) {
                    fprintf(stderr, "Incorrect argument\n");
                    return 1;
                }
                arguments.number_of_realisation = conv;
            } else {
                fprintf(stderr, "Incorrect argument\n");
                return 1;
            }
        }
    }
    if (arguments.input_file_name == NULL) {
        fprintf(stderr, "Put input file name\n");
        return 1;
    }
    if (arguments.output_file_name == NULL) {
        fprintf(stderr, "Put output file name\n");
        return 1;
    }
    if (arguments.number_of_realisation == -1) {
        fprintf(stderr, "Put number of realisation\n");
        return 1;
    }
    size_t n, k, m;
    FILE *in = fopen(arguments.input_file_name, "rb");
    if (in == NULL) {
        fprintf(stderr, "Input file is not open");
        return 1;
    }
    if (fscanf(in, "%zu %zu %zu", &n, &k, &m) != 3) {
        fprintf(stderr, "Incorrect format of data");
        return 1;
    }
    cl_float *a = malloc(k * m * sizeof(cl_float));
    if (a == NULL) {
        fprintf(stderr, "Unable to allocate memory");
        return 1;
    }
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < k; j++) {
            if (fscanf(in, "%f", &a[i * k + j]) != 1) {
                free(a);
                fprintf(stderr, "Error while reading");
                return 3;
            }
        }
    }
    cl_float *b = malloc(n * k * sizeof(cl_float));
    if (b == NULL) {
        free(a);
        fprintf(stderr, "Unable to allocate memory");
        return 3;
    }
    for (size_t i = 0; i < k; i++) {
        for (size_t j = 0; j < n; j++) {
            if (fscanf(in, "%f", &b[i * n + j]) != 1) {
                free(a);
                free(b);
                fprintf(stderr, "Error while reading");
                return 1;
            }
        }
    }
    cl_device_id current_device = NULL;
    if (arguments.number_of_realisation != 0) {
        current_device = get_platforms(&arguments);
        if (current_device != NULL) {
            size_t size_name;
            clGetDeviceInfo(current_device, CL_DEVICE_NAME, 0, NULL, &size_name);
            char *device_name = malloc(sizeof(char) * size_name);
            if (device_name == NULL) {
                free(a);
                free(b);
                fprintf(stderr, "Unable to allocate memory");
                return 3;
            }
            clGetDeviceInfo(current_device, CL_DEVICE_NAME, size_name, device_name, NULL);
            cl_platform_id platform;
            clGetDeviceInfo(current_device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);
            size_t platform_name_size;
            clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &platform_name_size);
            char *platform_name = malloc(platform_name_size);
            if (platform_name == NULL) {
                free(a);
                free(b);
                free(device_name);
                fprintf(stderr, "Unable to allocate memory");
                return 3;
            }
            clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name_size, platform_name, NULL);
            printf("Device: %s\tPlatform: %s\n", device_name, platform_name);
            free(device_name);
            free(platform_name);
        } else {
            free(a);
            free(b);
            fprintf(stderr, "No devices found");
            return 3;
        }
    }
    cl_float *c = malloc(n * m * sizeof(cl_float));
    if (c == NULL) {
        free(a);
        free(b);
        fprintf(stderr, "Unable to allocate memory");
        return 3;
    }
    switch (arguments.number_of_realisation) {
        case (0):
            calculate0(a, b, c, n, m, k);
            break;
        case(1):
            if (calculate1(current_device, a, b, c, n, m, k) != 0) {
                free(a);
                free(b);
                free(c);
                fprintf(stderr, "Error while running");
                return 3;
            }
            break;
        case(2):
            if (calculate2(current_device, a, b, c, n, m, k) != 0) {
                free(a);
                free(b);
                free(c);
                fprintf(stderr, "Error while running");
                return 3;
            }
            break;
        case(3):
            if (calculate3(current_device, a, b, c, n, m, k) != 0)  {
                free(a);
                free(b);
                free(c);
                fprintf(stderr, "Error while running");
                return 3;
            }
            break;
    }
    free(a);
    free(b);
    int output_err = print_output(&arguments, n, m, c);
    free(c);

    return output_err;

}

int print_output(struct arguments *arguments, size_t n, size_t m, const cl_float *c) {
    FILE *out = fopen((*arguments).output_file_name, "wb");
    if (out == NULL) {
        fprintf(stderr, "Output file is not open");
        return 1;
    }
    if (fprintf(out, "%zu %zu\n", n, m) < 0) {
        fprintf(stderr, "Error while writing");
        return 1;
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fprintf(out, "%f ", c[i * n + j]) < 0) {
                fprintf(stderr, "Error while writing");
                return 1;
            }
        }
        if (fprintf(out, "\n") < 0) {
            fprintf(stderr, "Error while writing");
            return 1;
        }
    }
    fclose(out);
    return 0;
}
