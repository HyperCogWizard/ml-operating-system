/*
 * GGUF Integration Header
 * 
 * Defines GGUF serialization and deserialization functions
 */

#ifndef GGUF_INTEGRATION_H
#define GGUF_INTEGRATION_H

#include "robotics_middleware.h"
#include "tensor_abstraction.h"

/* GGUF-specific functions */
int gguf_validate_tensor(const tensor_t *tensor);
int gguf_get_tensor_metadata(const tensor_t *tensor, char *metadata, size_t size);
int gguf_set_tensor_metadata(tensor_t *tensor, const char *metadata);

#endif /* GGUF_INTEGRATION_H */