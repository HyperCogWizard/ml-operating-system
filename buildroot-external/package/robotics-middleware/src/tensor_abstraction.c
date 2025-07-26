/*
 * Tensor Abstraction for Robotics Middleware
 * 
 * Implements tensor operations with support for device/sensor/actuator
 * data representation and GGUF serialization.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>

#include "tensor_abstraction.h"
#include "robotics_middleware.h"

/* Global tensor system state */
static tensor_system_t *g_tensor_sys = NULL;

/*
 * Calculate tensor size in bytes based on specification
 */
uint64_t tensor_size_bytes(const tensor_spec_t *spec) {
    if (!spec || spec->dimensions == 0) {
        return 0;
    }
    
    uint64_t total_elements = 1;
    for (uint32_t i = 0; i < spec->dimensions && i < 8; i++) {
        if (spec->shape[i] == 0) {
            return 0;
        }
        total_elements *= spec->shape[i];
    }
    
    uint32_t element_size = 0;
    switch (spec->dtype) {
        case TENSOR_UINT8:
        case TENSOR_INT8:
            element_size = 1;
            break;
        case TENSOR_UINT16:
        case TENSOR_INT16:
            element_size = 2;
            break;
        case TENSOR_UINT32:
        case TENSOR_INT32:
        case TENSOR_FLOAT32:
            element_size = 4;
            break;
        case TENSOR_FLOAT64:
            element_size = 8;
            break;
        default:
            return 0;
    }
    
    return total_elements * element_size;
}

/*
 * Create a new tensor with the given specification
 */
tensor_t* tensor_create(const tensor_spec_t *spec) {
    if (!spec) {
        return NULL;
    }
    
    uint64_t size = tensor_size_bytes(spec);
    if (size == 0) {
        return NULL;
    }
    
    tensor_t *tensor = calloc(1, sizeof(tensor_t));
    if (!tensor) {
        return NULL;
    }
    
    tensor->data = calloc(1, size);
    if (!tensor->data) {
        free(tensor);
        return NULL;
    }
    
    tensor->spec = *spec;
    tensor->size_bytes = size;
    tensor->ref_count = 1;
    
    if (pthread_mutex_init(&tensor->mutex, NULL) != 0) {
        free(tensor->data);
        free(tensor);
        return NULL;
    }
    
    /* Update global tensor system stats */
    if (g_tensor_sys) {
        pthread_mutex_lock(&g_tensor_sys->alloc_mutex);
        g_tensor_sys->used_memory += size;
        g_tensor_sys->tensor_count++;
        pthread_mutex_unlock(&g_tensor_sys->alloc_mutex);
    }
    
    return tensor;
}

/*
 * Destroy a tensor and free its memory
 */
void tensor_destroy(tensor_t *tensor) {
    if (!tensor) {
        return;
    }
    
    pthread_mutex_lock(&tensor->mutex);
    
    if (--tensor->ref_count > 0) {
        pthread_mutex_unlock(&tensor->mutex);
        return;
    }
    
    /* Update global tensor system stats */
    if (g_tensor_sys) {
        pthread_mutex_lock(&g_tensor_sys->alloc_mutex);
        g_tensor_sys->used_memory -= tensor->size_bytes;
        g_tensor_sys->tensor_count--;
        pthread_mutex_unlock(&g_tensor_sys->alloc_mutex);
    }
    
    pthread_mutex_unlock(&tensor->mutex);
    pthread_mutex_destroy(&tensor->mutex);
    
    free(tensor->data);
    free(tensor);
}

/*
 * Copy tensor data from source to destination
 */
int tensor_copy(tensor_t *dst, const tensor_t *src) {
    if (!dst || !src) {
        return -EINVAL;
    }
    
    if (dst->size_bytes != src->size_bytes) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&dst->mutex);
    pthread_mutex_lock((pthread_mutex_t*)&src->mutex);
    
    memcpy(dst->data, src->data, src->size_bytes);
    dst->spec = src->spec;
    
    pthread_mutex_unlock((pthread_mutex_t*)&src->mutex);
    pthread_mutex_unlock(&dst->mutex);
    
    return 0;
}

/*
 * Initialize tensor system
 */
int tensor_system_init(tensor_system_t *sys) {
    if (!sys) {
        return -EINVAL;
    }
    
    memset(sys, 0, sizeof(tensor_system_t));
    
    if (pthread_mutex_init(&sys->alloc_mutex, NULL) != 0) {
        return -EINVAL;
    }
    
    sys->total_memory = 1024 * 1024 * 1024; /* 1GB default limit */
    g_tensor_sys = sys;
    
    return 0;
}

/*
 * Cleanup tensor system
 */
void tensor_system_cleanup(tensor_system_t *sys) {
    if (!sys) {
        return;
    }
    
    pthread_mutex_destroy(&sys->alloc_mutex);
    g_tensor_sys = NULL;
}

/*
 * Get current memory usage
 */
uint64_t tensor_system_memory_usage(const tensor_system_t *sys) {
    if (!sys) {
        return 0;
    }
    
    return sys->used_memory;
}

/*
 * Set tensor data from raw buffer
 */
int tensor_set_data(tensor_t *tensor, const void *data, uint64_t size) {
    if (!tensor || !data) {
        return -EINVAL;
    }
    
    if (size > tensor->size_bytes) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&tensor->mutex);
    memcpy(tensor->data, data, size);
    pthread_mutex_unlock(&tensor->mutex);
    
    return 0;
}

/*
 * Get tensor data to raw buffer
 */
int tensor_get_data(const tensor_t *tensor, void *data, uint64_t size) {
    if (!tensor || !data) {
        return -EINVAL;
    }
    
    if (size < tensor->size_bytes) {
        return -EINVAL;
    }
    
    pthread_mutex_lock((pthread_mutex_t*)&tensor->mutex);
    memcpy(data, tensor->data, tensor->size_bytes);
    pthread_mutex_unlock((pthread_mutex_t*)&tensor->mutex);
    
    return 0;
}

/*
 * Tensor element-wise operations
 */
int tensor_add(tensor_t *result, const tensor_t *a, const tensor_t *b) {
    if (!result || !a || !b) {
        return -EINVAL;
    }
    
    if (a->size_bytes != b->size_bytes || a->size_bytes != result->size_bytes) {
        return -EINVAL;
    }
    
    if (a->spec.dtype != b->spec.dtype || a->spec.dtype != result->spec.dtype) {
        return -EINVAL;
    }
    
    pthread_mutex_lock((pthread_mutex_t*)&a->mutex);
    pthread_mutex_lock((pthread_mutex_t*)&b->mutex);
    pthread_mutex_lock(&result->mutex);
    
    uint64_t elements = a->size_bytes;
    
    switch (a->spec.dtype) {
        case TENSOR_FLOAT32: {
            float *ra = (float*)result->data;
            float *aa = (float*)a->data;
            float *ba = (float*)b->data;
            elements /= sizeof(float);
            for (uint64_t i = 0; i < elements; i++) {
                ra[i] = aa[i] + ba[i];
            }
            break;
        }
        case TENSOR_FLOAT64: {
            double *ra = (double*)result->data;
            double *aa = (double*)a->data;
            double *ba = (double*)b->data;
            elements /= sizeof(double);
            for (uint64_t i = 0; i < elements; i++) {
                ra[i] = aa[i] + ba[i];
            }
            break;
        }
        case TENSOR_INT32: {
            int32_t *ra = (int32_t*)result->data;
            int32_t *aa = (int32_t*)a->data;
            int32_t *ba = (int32_t*)b->data;
            elements /= sizeof(int32_t);
            for (uint64_t i = 0; i < elements; i++) {
                ra[i] = aa[i] + ba[i];
            }
            break;
        }
        default:
            pthread_mutex_unlock(&result->mutex);
            pthread_mutex_unlock((pthread_mutex_t*)&b->mutex);
            pthread_mutex_unlock((pthread_mutex_t*)&a->mutex);
            return -ENOTSUP;
    }
    
    pthread_mutex_unlock(&result->mutex);
    pthread_mutex_unlock((pthread_mutex_t*)&b->mutex);
    pthread_mutex_unlock((pthread_mutex_t*)&a->mutex);
    
    return 0;
}

/*
 * Initialize tensor with zeros
 */
int tensor_zeros(tensor_t *tensor) {
    if (!tensor) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&tensor->mutex);
    memset(tensor->data, 0, tensor->size_bytes);
    pthread_mutex_unlock(&tensor->mutex);
    
    return 0;
}

/*
 * Initialize tensor with random values
 */
int tensor_random(tensor_t *tensor) {
    if (!tensor) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&tensor->mutex);
    
    uint64_t elements = tensor->size_bytes;
    
    switch (tensor->spec.dtype) {
        case TENSOR_FLOAT32: {
            float *data = (float*)tensor->data;
            elements /= sizeof(float);
            for (uint64_t i = 0; i < elements; i++) {
                data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            }
            break;
        }
        case TENSOR_FLOAT64: {
            double *data = (double*)tensor->data;
            elements /= sizeof(double);
            for (uint64_t i = 0; i < elements; i++) {
                data[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }
            break;
        }
        default:
            /* Fill with random bytes for other types */
            for (uint64_t i = 0; i < tensor->size_bytes; i++) {
                ((uint8_t*)tensor->data)[i] = rand() & 0xFF;
            }
            break;
    }
    
    pthread_mutex_unlock(&tensor->mutex);
    
    return 0;
}

/*
 * Print tensor information for debugging
 */
void tensor_print_info(const tensor_t *tensor) {
    if (!tensor) {
        printf("Tensor: NULL\n");
        return;
    }
    
    printf("Tensor Info:\n");
    printf("  Dimensions: %u\n", tensor->spec.dimensions);
    printf("  Shape: [");
    for (uint32_t i = 0; i < tensor->spec.dimensions && i < 8; i++) {
        printf("%u", tensor->spec.shape[i]);
        if (i < tensor->spec.dimensions - 1) printf(", ");
    }
    printf("]\n");
    printf("  Data type: %d\n", tensor->spec.dtype);
    printf("  Size: %lu bytes\n", tensor->size_bytes);
    printf("  Ref count: %u\n", tensor->ref_count);
    printf("  Metadata: %s\n", tensor->spec.metadata);
}