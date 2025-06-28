/*
 * Tensor Abstraction Header
 * 
 * Defines tensor operations and data structures for robotics middleware
 */

#ifndef TENSOR_ABSTRACTION_H
#define TENSOR_ABSTRACTION_H

#include "robotics_middleware.h"

/* Tensor operations */
int tensor_set_data(tensor_t *tensor, const void *data, uint64_t size);
int tensor_get_data(const tensor_t *tensor, void *data, uint64_t size);
int tensor_add(tensor_t *result, const tensor_t *a, const tensor_t *b);
int tensor_zeros(tensor_t *tensor);
int tensor_random(tensor_t *tensor);
void tensor_print_info(const tensor_t *tensor);

/* Tensor reference counting */
static inline void tensor_ref(tensor_t *tensor) {
    if (tensor) {
        pthread_mutex_lock(&tensor->mutex);
        tensor->ref_count++;
        pthread_mutex_unlock(&tensor->mutex);
    }
}

static inline void tensor_unref(tensor_t *tensor) {
    if (tensor) {
        tensor_destroy(tensor);
    }
}

#endif /* TENSOR_ABSTRACTION_H */