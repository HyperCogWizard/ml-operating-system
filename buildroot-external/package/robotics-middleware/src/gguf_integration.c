/*
 * GGUF Integration for Robotics Middleware
 * 
 * Implements GGUF serialization and deserialization for agent states,
 * device configurations, and tensor data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <endian.h>

#include "gguf_integration.h"
#include "robotics_middleware.h"

/* GGUF file format constants */
#define GGUF_MAGIC 0x46554747  /* "GGUF" */
#define GGUF_VERSION 3
#define GGUF_DEFAULT_ALIGNMENT 32

/* GGUF data types */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type_t;

/* GGUF header structure */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} gguf_header_t;

/* GGUF tensor info */
typedef struct {
    char name[MAX_NAME_LEN];
    uint32_t n_dimensions;
    uint64_t dimensions[8];
    gguf_type_t type;
    uint64_t offset;
} gguf_tensor_info_t;

/*
 * Convert tensor dtype to GGUF type
 */
static gguf_type_t tensor_dtype_to_gguf(tensor_dtype_t dtype) {
    switch (dtype) {
        case TENSOR_UINT8:  return GGUF_TYPE_UINT8;
        case TENSOR_INT8:   return GGUF_TYPE_INT8;
        case TENSOR_UINT16: return GGUF_TYPE_UINT16;
        case TENSOR_INT16:  return GGUF_TYPE_INT16;
        case TENSOR_UINT32: return GGUF_TYPE_UINT32;
        case TENSOR_INT32:  return GGUF_TYPE_INT32;
        case TENSOR_FLOAT32: return GGUF_TYPE_FLOAT32;
        case TENSOR_FLOAT64: return GGUF_TYPE_FLOAT64;
        default: return GGUF_TYPE_UINT8;
    }
}

/*
 * Convert GGUF type to tensor dtype
 */
static tensor_dtype_t gguf_type_to_tensor_dtype(gguf_type_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  return TENSOR_UINT8;
        case GGUF_TYPE_INT8:   return TENSOR_INT8;
        case GGUF_TYPE_UINT16: return TENSOR_UINT16;
        case GGUF_TYPE_INT16:  return TENSOR_INT16;
        case GGUF_TYPE_UINT32: return TENSOR_UINT32;
        case GGUF_TYPE_INT32:  return TENSOR_INT32;
        case GGUF_TYPE_FLOAT32: return TENSOR_FLOAT32;
        case GGUF_TYPE_FLOAT64: return TENSOR_FLOAT64;
        default: return TENSOR_UINT8;
    }
}

/*
 * Initialize GGUF integration
 */
int gguf_integration_init(gguf_context_t *ctx) {
    if (!ctx) {
        return -EINVAL;
    }
    
    memset(ctx, 0, sizeof(gguf_context_t));
    strncpy(ctx->version, "1.0.0", sizeof(ctx->version) - 1);
    ctx->initialized = true;
    
    return 0;
}

/*
 * Cleanup GGUF integration
 */
void gguf_integration_cleanup(gguf_context_t *ctx) {
    if (!ctx) {
        return;
    }
    
    if (ctx->metadata) {
        free(ctx->metadata);
    }
    
    memset(ctx, 0, sizeof(gguf_context_t));
}

/*
 * Prepare export context with all tensors from the system
 */
int gguf_prepare_export(const robotics_context_t *ctx, gguf_export_context_t *export_ctx) {
    if (!ctx || !export_ctx) {
        return -EINVAL;
    }
    
    memset(export_ctx, 0, sizeof(gguf_export_context_t));
    
    /* Count total tensors from devices and agents */
    uint32_t total_tensors = 0;
    
    pthread_rwlock_rdlock((pthread_rwlock_t*)&ctx->hypergraph->lock);
    
    for (uint32_t i = 0; i < ctx->hypergraph->device_count; i++) {
        if (ctx->hypergraph->devices[i] && ctx->hypergraph->devices[i]->tensor) {
            total_tensors++;
        }
    }
    
    for (uint32_t i = 0; i < ctx->hypergraph->agent_count; i++) {
        if (ctx->hypergraph->agents[i] && ctx->hypergraph->agents[i]->cognitive_tensor) {
            total_tensors++;
        }
    }
    
    if (total_tensors == 0) {
        pthread_rwlock_unlock((pthread_rwlock_t*)&ctx->hypergraph->lock);
        return 0;
    }
    
    /* Allocate arrays */
    export_ctx->tensors = calloc(total_tensors, sizeof(tensor_t*));
    export_ctx->tensor_names = calloc(total_tensors, sizeof(char*));
    
    if (!export_ctx->tensors || !export_ctx->tensor_names) {
        pthread_rwlock_unlock((pthread_rwlock_t*)&ctx->hypergraph->lock);
        gguf_cleanup_export(export_ctx);
        return -ENOMEM;
    }
    
    /* Collect tensors */
    uint32_t tensor_idx = 0;
    
    /* Device tensors */
    for (uint32_t i = 0; i < ctx->hypergraph->device_count; i++) {
        device_node_t *device = ctx->hypergraph->devices[i];
        if (device && device->tensor) {
            export_ctx->tensors[tensor_idx] = device->tensor;
            export_ctx->tensor_names[tensor_idx] = malloc(MAX_NAME_LEN);
            snprintf(export_ctx->tensor_names[tensor_idx], MAX_NAME_LEN, 
                    "device_%u_%s", device->id, device->name);
            tensor_idx++;
        }
    }
    
    /* Agent tensors */
    for (uint32_t i = 0; i < ctx->hypergraph->agent_count; i++) {
        agent_t *agent = ctx->hypergraph->agents[i];
        if (agent && agent->cognitive_tensor) {
            export_ctx->tensors[tensor_idx] = agent->cognitive_tensor;
            export_ctx->tensor_names[tensor_idx] = malloc(MAX_NAME_LEN);
            snprintf(export_ctx->tensor_names[tensor_idx], MAX_NAME_LEN, 
                    "agent_%u_%s", agent->id, agent->name);
            tensor_idx++;
        }
    }
    
    export_ctx->tensor_count = tensor_idx;
    
    pthread_rwlock_unlock((pthread_rwlock_t*)&ctx->hypergraph->lock);
    
    printf("Prepared %u tensors for GGUF export\n", export_ctx->tensor_count);
    return 0;
}

/*
 * Write GGUF file
 */
int gguf_write_file(const gguf_export_context_t *export_ctx, const char *filepath) {
    if (!export_ctx || !filepath) {
        return -EINVAL;
    }
    
    FILE *file = fopen(filepath, "wb");
    if (!file) {
        return -errno;
    }
    
    /* Write header */
    gguf_header_t header = {
        .magic = htole32(GGUF_MAGIC),
        .version = htole32(GGUF_VERSION),
        .tensor_count = htole64(export_ctx->tensor_count),
        .metadata_kv_count = htole64(0)  /* No metadata for now */
    };
    
    if (fwrite(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        return -EIO;
    }
    
    /* Write tensor info */
    uint64_t data_offset = sizeof(header);
    data_offset += export_ctx->tensor_count * sizeof(gguf_tensor_info_t);
    
    /* Align data offset */
    data_offset = (data_offset + GGUF_DEFAULT_ALIGNMENT - 1) & 
                  ~(GGUF_DEFAULT_ALIGNMENT - 1);
    
    uint64_t current_offset = data_offset;
    
    for (uint32_t i = 0; i < export_ctx->tensor_count; i++) {
        tensor_t *tensor = export_ctx->tensors[i];
        
        gguf_tensor_info_t info = {0};
        strncpy(info.name, export_ctx->tensor_names[i], sizeof(info.name) - 1);
        info.n_dimensions = htole32(tensor->spec.dimensions);
        
        for (uint32_t j = 0; j < tensor->spec.dimensions && j < 8; j++) {
            info.dimensions[j] = htole64(tensor->spec.shape[j]);
        }
        
        info.type = htole32(tensor_dtype_to_gguf(tensor->spec.dtype));
        info.offset = htole64(current_offset);
        
        if (fwrite(&info, sizeof(info), 1, file) != 1) {
            fclose(file);
            return -EIO;
        }
        
        current_offset += tensor->size_bytes;
        /* Align next tensor */
        current_offset = (current_offset + GGUF_DEFAULT_ALIGNMENT - 1) & 
                        ~(GGUF_DEFAULT_ALIGNMENT - 1);
    }
    
    /* Seek to data section */
    if (fseek(file, data_offset, SEEK_SET) != 0) {
        fclose(file);
        return -EIO;
    }
    
    /* Write tensor data */
    for (uint32_t i = 0; i < export_ctx->tensor_count; i++) {
        tensor_t *tensor = export_ctx->tensors[i];
        
        pthread_mutex_lock((pthread_mutex_t*)&tensor->mutex);
        
        if (fwrite(tensor->data, tensor->size_bytes, 1, file) != 1) {
            pthread_mutex_unlock((pthread_mutex_t*)&tensor->mutex);
            fclose(file);
            return -EIO;
        }
        
        pthread_mutex_unlock((pthread_mutex_t*)&tensor->mutex);
        
        /* Write alignment padding */
        uint64_t padding = ((tensor->size_bytes + GGUF_DEFAULT_ALIGNMENT - 1) & 
                           ~(GGUF_DEFAULT_ALIGNMENT - 1)) - tensor->size_bytes;
        
        if (padding > 0) {
            uint8_t zeros[GGUF_DEFAULT_ALIGNMENT] = {0};
            if (fwrite(zeros, padding, 1, file) != 1) {
                fclose(file);
                return -EIO;
            }
        }
    }
    
    fclose(file);
    
    printf("Successfully wrote GGUF file: %s\n", filepath);
    return 0;
}

/*
 * Read GGUF file
 */
int gguf_read_file(gguf_import_context_t *import_ctx, const char *filepath) {
    if (!import_ctx || !filepath) {
        return -EINVAL;
    }
    
    memset(import_ctx, 0, sizeof(gguf_import_context_t));
    
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        return -errno;
    }
    
    /* Read header */
    gguf_header_t header;
    if (fread(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        return -EIO;
    }
    
    /* Verify magic and version */
    if (le32toh(header.magic) != GGUF_MAGIC) {
        fclose(file);
        return -EINVAL;
    }
    
    if (le32toh(header.version) != GGUF_VERSION) {
        fclose(file);
        return -ENOTSUP;
    }
    
    import_ctx->tensor_count = le64toh(header.tensor_count);
    
    if (import_ctx->tensor_count == 0) {
        fclose(file);
        return 0;
    }
    
    /* Allocate arrays */
    import_ctx->tensors = calloc(import_ctx->tensor_count, sizeof(tensor_t*));
    import_ctx->tensor_names = calloc(import_ctx->tensor_count, sizeof(char*));
    
    if (!import_ctx->tensors || !import_ctx->tensor_names) {
        fclose(file);
        gguf_cleanup_import(import_ctx);
        return -ENOMEM;
    }
    
    /* Read tensor info and create tensors */
    for (uint32_t i = 0; i < import_ctx->tensor_count; i++) {
        gguf_tensor_info_t info;
        if (fread(&info, sizeof(info), 1, file) != 1) {
            fclose(file);
            gguf_cleanup_import(import_ctx);
            return -EIO;
        }
        
        /* Create tensor spec */
        tensor_spec_t spec = {0};
        spec.dimensions = le32toh(info.n_dimensions);
        spec.dtype = gguf_type_to_tensor_dtype(le32toh(info.type));
        
        for (uint32_t j = 0; j < spec.dimensions && j < 8; j++) {
            spec.shape[j] = le64toh(info.dimensions[j]);
        }
        
        /* Create tensor */
        tensor_t *tensor = tensor_create(&spec);
        if (!tensor) {
            fclose(file);
            gguf_cleanup_import(import_ctx);
            return -ENOMEM;
        }
        
        import_ctx->tensors[i] = tensor;
        import_ctx->tensor_names[i] = malloc(MAX_NAME_LEN);
        strncpy(import_ctx->tensor_names[i], info.name, MAX_NAME_LEN - 1);
    }
    
    /* Read tensor data */
    uint64_t data_offset = sizeof(header);
    data_offset += import_ctx->tensor_count * sizeof(gguf_tensor_info_t);
    data_offset = (data_offset + GGUF_DEFAULT_ALIGNMENT - 1) & 
                  ~(GGUF_DEFAULT_ALIGNMENT - 1);
    
    if (fseek(file, data_offset, SEEK_SET) != 0) {
        fclose(file);
        gguf_cleanup_import(import_ctx);
        return -EIO;
    }
    
    for (uint32_t i = 0; i < import_ctx->tensor_count; i++) {
        tensor_t *tensor = import_ctx->tensors[i];
        
        if (fread(tensor->data, tensor->size_bytes, 1, file) != 1) {
            fclose(file);
            gguf_cleanup_import(import_ctx);
            return -EIO;
        }
        
        /* Skip alignment padding */
        uint64_t padding = ((tensor->size_bytes + GGUF_DEFAULT_ALIGNMENT - 1) & 
                           ~(GGUF_DEFAULT_ALIGNMENT - 1)) - tensor->size_bytes;
        
        if (padding > 0 && fseek(file, padding, SEEK_CUR) != 0) {
            fclose(file);
            gguf_cleanup_import(import_ctx);
            return -EIO;
        }
    }
    
    fclose(file);
    
    printf("Successfully read GGUF file: %s (%u tensors)\n", 
           filepath, import_ctx->tensor_count);
    return 0;
}

/*
 * Restore system state from import context
 */
int gguf_restore_system(robotics_context_t *ctx, const gguf_import_context_t *import_ctx) {
    if (!ctx || !import_ctx) {
        return -EINVAL;
    }
    
    /* For now, just print tensor information */
    /* TODO: Implement proper restoration of devices and agents */
    
    printf("Restoring system from GGUF data:\n");
    for (uint32_t i = 0; i < import_ctx->tensor_count; i++) {
        printf("  Tensor %u: %s\n", i, import_ctx->tensor_names[i]);
        tensor_print_info(import_ctx->tensors[i]);
    }
    
    return 0;
}

/*
 * Cleanup export context
 */
void gguf_cleanup_export(gguf_export_context_t *export_ctx) {
    if (!export_ctx) {
        return;
    }
    
    if (export_ctx->tensor_names) {
        for (uint32_t i = 0; i < export_ctx->tensor_count; i++) {
            free(export_ctx->tensor_names[i]);
        }
        free(export_ctx->tensor_names);
    }
    
    free(export_ctx->tensors);
    
    if (export_ctx->metadata) {
        free(export_ctx->metadata);
    }
    
    memset(export_ctx, 0, sizeof(gguf_export_context_t));
}

/*
 * Cleanup import context
 */
void gguf_cleanup_import(gguf_import_context_t *import_ctx) {
    if (!import_ctx) {
        return;
    }
    
    if (import_ctx->tensors) {
        for (uint32_t i = 0; i < import_ctx->tensor_count; i++) {
            if (import_ctx->tensors[i]) {
                tensor_destroy(import_ctx->tensors[i]);
            }
        }
        free(import_ctx->tensors);
    }
    
    if (import_ctx->tensor_names) {
        for (uint32_t i = 0; i < import_ctx->tensor_count; i++) {
            free(import_ctx->tensor_names[i]);
        }
        free(import_ctx->tensor_names);
    }
    
    if (import_ctx->metadata) {
        free(import_ctx->metadata);
    }
    
    memset(import_ctx, 0, sizeof(gguf_import_context_t));
}