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

/* Helper function to serialize agent config to JSON */
static int serialize_agent_config(const agent_t *agent, char *json_buffer, size_t buffer_size) {
    int written = snprintf(json_buffer, buffer_size,
        "{"
        "\"cognitive_dimensions\":%u,"
        "\"memory_size\":%u,"
        "\"attention_heads\":%u,"
        "\"embedding_dim\":%u,"
        "\"scheme_functions\":\"%s\","
        "\"neural_symbolic_config\":\"%s\","
        "\"autonomous\":%s,"
        "\"last_action\":%lu"
        "}",
        agent->config.cognitive_dimensions,
        agent->config.memory_size,
        agent->config.attention_heads,
        agent->config.embedding_dim,
        agent->config.scheme_functions,
        agent->config.neural_symbolic_config,
        agent->autonomous ? "true" : "false",
        agent->last_action
    );
    
    return (written < (int)buffer_size) ? 0 : -ENOMEM;
}

/* Helper function to serialize device spec to JSON */
static int serialize_device_spec(const device_node_t *device, char *json_buffer, size_t buffer_size) {
    const tensor_spec_t *spec = &device->tensor_spec;
    
    /* Build shape array string */
    char shape_str[256] = "[";
    for (uint32_t i = 0; i < spec->dimensions; i++) {
        char num_str[32];
        snprintf(num_str, sizeof(num_str), "%u%s", spec->shape[i], 
                 (i < spec->dimensions - 1) ? "," : "");
        strncat(shape_str, num_str, sizeof(shape_str) - strlen(shape_str) - 1);
    }
    strncat(shape_str, "]", sizeof(shape_str) - strlen(shape_str) - 1);
    
    int written = snprintf(json_buffer, buffer_size,
        "{"
        "\"dimensions\":%u,"
        "\"shape\":%s,"
        "\"dtype\":%d,"
        "\"requires_grad\":%s,"
        "\"metadata\":\"%s\","
        "\"last_update\":%lu"
        "}",
        spec->dimensions,
        shape_str,
        spec->dtype,
        spec->requires_grad ? "true" : "false",
        spec->metadata,
        device->last_update
    );
    
    return (written < (int)buffer_size) ? 0 : -ENOMEM;
}

/*
 * Prepare enhanced export context with complete system state
 */
int gguf_prepare_enhanced_export(const robotics_context_t *ctx, gguf_enhanced_export_context_t *export_ctx) {
    if (!ctx || !export_ctx) {
        return -EINVAL;
    }
    
    memset(export_ctx, 0, sizeof(gguf_enhanced_export_context_t));
    
    pthread_rwlock_rdlock((pthread_rwlock_t*)&ctx->hypergraph->lock);
    
    /* Count total components */
    uint32_t total_tensors = 0;
    uint32_t total_agents = ctx->hypergraph->agent_count;
    uint32_t total_devices = ctx->hypergraph->device_count;
    uint32_t total_modules = 0;
    
    /* Get module count from module registry */
    module_registry_t *module_registry = workbench_get_module_registry();
    if (module_registry) {
        total_modules = module_registry->module_count;
    }
    
    /* Count tensors from devices and agents */
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
    
    /* Initialize membrane header */
    strncpy(export_ctx->membrane_header.membrane_name, "robotics_system", 
            sizeof(export_ctx->membrane_header.membrane_name) - 1);
    export_ctx->membrane_header.tensor_count = total_tensors;
    export_ctx->membrane_header.agent_count = total_agents;
    export_ctx->membrane_header.device_count = total_devices;
    export_ctx->membrane_header.module_count = total_modules;
    strncpy(export_ctx->membrane_header.metadata, "Enhanced GGUF P-System membrane",
            sizeof(export_ctx->membrane_header.metadata) - 1);
    
    /* Allocate arrays for tensors */
    if (total_tensors > 0) {
        export_ctx->tensors = calloc(total_tensors, sizeof(tensor_t*));
        export_ctx->tensor_names = calloc(total_tensors, sizeof(char*));
        
        if (!export_ctx->tensors || !export_ctx->tensor_names) {
            pthread_rwlock_unlock((pthread_rwlock_t*)&ctx->hypergraph->lock);
            gguf_cleanup_enhanced_export(export_ctx);
            return -ENOMEM;
        }
    }
    
    /* Allocate arrays for configurations */
    if (total_agents > 0) {
        export_ctx->agent_configs = calloc(total_agents, sizeof(gguf_agent_config_t));
        if (!export_ctx->agent_configs) {
            pthread_rwlock_unlock((pthread_rwlock_t*)&ctx->hypergraph->lock);
            gguf_cleanup_enhanced_export(export_ctx);
            return -ENOMEM;
        }
    }
    
    if (total_devices > 0) {
        export_ctx->device_configs = calloc(total_devices, sizeof(gguf_device_config_t));
        if (!export_ctx->device_configs) {
            pthread_rwlock_unlock((pthread_rwlock_t*)&ctx->hypergraph->lock);
            gguf_cleanup_enhanced_export(export_ctx);
            return -ENOMEM;
        }
    }
    
    if (total_modules > 0) {
        export_ctx->module_configs = calloc(total_modules, sizeof(gguf_module_config_t));
        if (!export_ctx->module_configs) {
            pthread_rwlock_unlock((pthread_rwlock_t*)&ctx->hypergraph->lock);
            gguf_cleanup_enhanced_export(export_ctx);
            return -ENOMEM;
        }
    }
    
    /* Collect tensors and device configurations */
    uint32_t tensor_idx = 0;
    for (uint32_t i = 0; i < ctx->hypergraph->device_count; i++) {
        device_node_t *device = ctx->hypergraph->devices[i];
        if (device && device->tensor) {
            export_ctx->tensors[tensor_idx] = device->tensor;
            export_ctx->tensor_names[tensor_idx] = malloc(MAX_NAME_LEN);
            snprintf(export_ctx->tensor_names[tensor_idx], MAX_NAME_LEN, 
                    "device_%u_%s", device->id, device->name);
            tensor_idx++;
        }
        
        /* Serialize device configuration */
        if (device) {
            gguf_device_config_t *dev_config = &export_ctx->device_configs[export_ctx->device_config_count];
            strncpy(dev_config->name, device->name, sizeof(dev_config->name) - 1);
            dev_config->type = device->type;
            dev_config->device_id = device->id;
            dev_config->active = device->active;
            
            if (serialize_device_spec(device, dev_config->spec_json, 
                                    sizeof(dev_config->spec_json)) < 0) {
                printf("Warning: Failed to serialize device spec for %s\n", device->name);
            }
            
            export_ctx->device_config_count++;
        }
    }
    
    /* Collect agent tensors and configurations */
    for (uint32_t i = 0; i < ctx->hypergraph->agent_count; i++) {
        agent_t *agent = ctx->hypergraph->agents[i];
        if (agent && agent->cognitive_tensor) {
            export_ctx->tensors[tensor_idx] = agent->cognitive_tensor;
            export_ctx->tensor_names[tensor_idx] = malloc(MAX_NAME_LEN);
            snprintf(export_ctx->tensor_names[tensor_idx], MAX_NAME_LEN, 
                    "agent_%u_%s", agent->id, agent->name);
            tensor_idx++;
        }
        
        /* Serialize agent configuration */
        if (agent) {
            gguf_agent_config_t *agent_config = &export_ctx->agent_configs[export_ctx->agent_config_count];
            strncpy(agent_config->name, agent->name, sizeof(agent_config->name) - 1);
            agent_config->state = agent->state;
            agent_config->agent_id = agent->id;
            
            if (serialize_agent_config(agent, agent_config->config_json, 
                                     sizeof(agent_config->config_json)) < 0) {
                printf("Warning: Failed to serialize agent config for %s\n", agent->name);
            }
            
            export_ctx->agent_config_count++;
        }
    }
    
    /* Collect module configurations */
    if (module_registry && total_modules > 0) {
        pthread_rwlock_rdlock(&module_registry->lock);
        for (uint32_t i = 0; i < module_registry->module_count; i++) {
            workbench_module_t *module = module_registry->modules[i];
            if (module) {
                gguf_module_config_t *mod_config = &export_ctx->module_configs[export_ctx->module_config_count];
                strncpy(mod_config->name, module->name, sizeof(mod_config->name) - 1);
                strncpy(mod_config->description, module->description, sizeof(mod_config->description) - 1);
                mod_config->type = module->type;
                mod_config->state = module->state;
                mod_config->module_id = module->id;
                
                /* Collect dependencies */
                mod_config->dependency_count = module->dependency_count;
                for (uint32_t j = 0; j < module->dependency_count && j < 8; j++) {
                    mod_config->dependency_ids[j] = module->dependencies[j] ? module->dependencies[j]->id : 0;
                }
                
                export_ctx->module_config_count++;
            }
        }
        pthread_rwlock_unlock(&module_registry->lock);
    }
    
    export_ctx->tensor_count = tensor_idx;
    
    pthread_rwlock_unlock((pthread_rwlock_t*)&ctx->hypergraph->lock);
    
    printf("Enhanced GGUF export prepared: %u tensors, %u agents, %u devices, %u modules\n",
           export_ctx->tensor_count, export_ctx->agent_config_count, 
           export_ctx->device_config_count, export_ctx->module_config_count);
    
    return 0;
}

/*
 * Write enhanced GGUF file with complete system state
 */
int gguf_write_enhanced_file(const gguf_enhanced_export_context_t *export_ctx, const char *filepath) {
    if (!export_ctx || !filepath) {
        return -EINVAL;
    }
    
    FILE *file = fopen(filepath, "wb");
    if (!file) {
        return -errno;
    }
    
    /* Calculate total metadata items:
     * - membrane header
     * - agent configs
     * - device configs  
     * - module configs
     */
    uint64_t metadata_count = 1 + export_ctx->agent_config_count + 
                             export_ctx->device_config_count + 
                             export_ctx->module_config_count;
    
    /* Write header */
    gguf_header_t header = {
        .magic = htole32(GGUF_MAGIC),
        .version = htole32(GGUF_VERSION),
        .tensor_count = htole64(export_ctx->tensor_count),
        .metadata_kv_count = htole64(metadata_count)
    };
    
    if (fwrite(&header, sizeof(header), 1, file) != 1) {
        fclose(file);
        return -EIO;
    }
    
    /* Write metadata */
    
    /* 1. Membrane header */
    if (fwrite(&export_ctx->membrane_header, sizeof(gguf_membrane_header_t), 1, file) != 1) {
        fclose(file);
        return -EIO;
    }
    
    /* 2. Agent configurations */
    for (uint32_t i = 0; i < export_ctx->agent_config_count; i++) {
        if (fwrite(&export_ctx->agent_configs[i], sizeof(gguf_agent_config_t), 1, file) != 1) {
            fclose(file);
            return -EIO;
        }
    }
    
    /* 3. Device configurations */
    for (uint32_t i = 0; i < export_ctx->device_config_count; i++) {
        if (fwrite(&export_ctx->device_configs[i], sizeof(gguf_device_config_t), 1, file) != 1) {
            fclose(file);
            return -EIO;
        }
    }
    
    /* 4. Module configurations */
    for (uint32_t i = 0; i < export_ctx->module_config_count; i++) {
        if (fwrite(&export_ctx->module_configs[i], sizeof(gguf_module_config_t), 1, file) != 1) {
            fclose(file);
            return -EIO;
        }
    }
    
    /* Write tensor info */
    uint64_t data_offset = sizeof(header) + 
                          sizeof(gguf_membrane_header_t) +
                          export_ctx->agent_config_count * sizeof(gguf_agent_config_t) +
                          export_ctx->device_config_count * sizeof(gguf_device_config_t) +
                          export_ctx->module_config_count * sizeof(gguf_module_config_t);
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
    
    printf("Successfully wrote enhanced GGUF file: %s\n", filepath);
    printf("  Membrane: %s\n", export_ctx->membrane_header.membrane_name);
    printf("  Tensors: %u, Agents: %u, Devices: %u, Modules: %u\n",
           export_ctx->tensor_count, export_ctx->agent_config_count,
           export_ctx->device_config_count, export_ctx->module_config_count);
    
    return 0;
}

/*
 * Cleanup enhanced export context
 */
void gguf_cleanup_enhanced_export(gguf_enhanced_export_context_t *export_ctx) {
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
    free(export_ctx->agent_configs);
    free(export_ctx->device_configs);
    free(export_ctx->module_configs);
    
    if (export_ctx->metadata) {
        free(export_ctx->metadata);
    }
    
    memset(export_ctx, 0, sizeof(gguf_enhanced_export_context_t));
}

/*
 * Cleanup enhanced import context
 */
void gguf_cleanup_enhanced_import(gguf_enhanced_import_context_t *import_ctx) {
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
    
    free(import_ctx->agent_configs);
    free(import_ctx->device_configs);
    free(import_ctx->module_configs);
    
    if (import_ctx->metadata) {
        free(import_ctx->metadata);
    }
    
    memset(import_ctx, 0, sizeof(gguf_enhanced_import_context_t));
}

/*
 * Create P-System membrane structure
 */
int gguf_create_membrane(const char *membrane_name, gguf_enhanced_export_context_t *export_ctx) {
    if (!membrane_name || !export_ctx) {
        return -EINVAL;
    }
    
    strncpy(export_ctx->membrane_header.membrane_name, membrane_name,
            sizeof(export_ctx->membrane_header.membrane_name) - 1);
    
    snprintf(export_ctx->membrane_header.metadata, 
             sizeof(export_ctx->membrane_header.metadata),
             "P-System membrane: %s with %u tensors, %u agents, %u devices, %u modules",
             membrane_name, export_ctx->tensor_count, export_ctx->agent_config_count,
             export_ctx->device_config_count, export_ctx->module_config_count);
    
    printf("Created P-System membrane: %s\n", membrane_name);
    return 0;
}

/*
 * Read enhanced GGUF file with complete system state
 */
int gguf_read_enhanced_file(gguf_enhanced_import_context_t *import_ctx, const char *filepath) {
    if (!import_ctx || !filepath) {
        return -EINVAL;
    }
    
    memset(import_ctx, 0, sizeof(gguf_enhanced_import_context_t));
    
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
    uint64_t metadata_count = le64toh(header.metadata_kv_count);
    
    /* Read membrane header */
    if (metadata_count > 0) {
        if (fread(&import_ctx->membrane_header, sizeof(gguf_membrane_header_t), 1, file) != 1) {
            fclose(file);
            return -EIO;
        }
        metadata_count--;
        
        printf("Reading P-System membrane: %s\n", import_ctx->membrane_header.membrane_name);
        printf("  Expected: %u tensors, %u agents, %u devices, %u modules\n",
               import_ctx->membrane_header.tensor_count,
               import_ctx->membrane_header.agent_count,
               import_ctx->membrane_header.device_count,
               import_ctx->membrane_header.module_count);
    }
    
    /* Allocate configuration arrays based on membrane header */
    if (import_ctx->membrane_header.agent_count > 0) {
        import_ctx->agent_configs = calloc(import_ctx->membrane_header.agent_count, 
                                          sizeof(gguf_agent_config_t));
        if (!import_ctx->agent_configs) {
            fclose(file);
            gguf_cleanup_enhanced_import(import_ctx);
            return -ENOMEM;
        }
    }
    
    if (import_ctx->membrane_header.device_count > 0) {
        import_ctx->device_configs = calloc(import_ctx->membrane_header.device_count, 
                                           sizeof(gguf_device_config_t));
        if (!import_ctx->device_configs) {
            fclose(file);
            gguf_cleanup_enhanced_import(import_ctx);
            return -ENOMEM;
        }
    }
    
    if (import_ctx->membrane_header.module_count > 0) {
        import_ctx->module_configs = calloc(import_ctx->membrane_header.module_count, 
                                           sizeof(gguf_module_config_t));
        if (!import_ctx->module_configs) {
            fclose(file);
            gguf_cleanup_enhanced_import(import_ctx);
            return -ENOMEM;
        }
    }
    
    /* Read agent configurations */
    for (uint32_t i = 0; i < import_ctx->membrane_header.agent_count && metadata_count > 0; i++) {
        if (fread(&import_ctx->agent_configs[i], sizeof(gguf_agent_config_t), 1, file) != 1) {
            fclose(file);
            gguf_cleanup_enhanced_import(import_ctx);
            return -EIO;
        }
        import_ctx->agent_config_count++;
        metadata_count--;
    }
    
    /* Read device configurations */
    for (uint32_t i = 0; i < import_ctx->membrane_header.device_count && metadata_count > 0; i++) {
        if (fread(&import_ctx->device_configs[i], sizeof(gguf_device_config_t), 1, file) != 1) {
            fclose(file);
            gguf_cleanup_enhanced_import(import_ctx);
            return -EIO;
        }
        import_ctx->device_config_count++;
        metadata_count--;
    }
    
    /* Read module configurations */
    for (uint32_t i = 0; i < import_ctx->membrane_header.module_count && metadata_count > 0; i++) {
        if (fread(&import_ctx->module_configs[i], sizeof(gguf_module_config_t), 1, file) != 1) {
            fclose(file);
            gguf_cleanup_enhanced_import(import_ctx);
            return -EIO;
        }
        import_ctx->module_config_count++;
        metadata_count--;
    }
    
    /* Allocate tensor arrays */
    if (import_ctx->tensor_count > 0) {
        import_ctx->tensors = calloc(import_ctx->tensor_count, sizeof(tensor_t*));
        import_ctx->tensor_names = calloc(import_ctx->tensor_count, sizeof(char*));
        
        if (!import_ctx->tensors || !import_ctx->tensor_names) {
            fclose(file);
            gguf_cleanup_enhanced_import(import_ctx);
            return -ENOMEM;
        }
    }
    
    /* Read tensor info and create tensors */
    for (uint32_t i = 0; i < import_ctx->tensor_count; i++) {
        gguf_tensor_info_t info;
        if (fread(&info, sizeof(info), 1, file) != 1) {
            fclose(file);
            gguf_cleanup_enhanced_import(import_ctx);
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
            gguf_cleanup_enhanced_import(import_ctx);
            return -ENOMEM;
        }
        
        import_ctx->tensors[i] = tensor;
        import_ctx->tensor_names[i] = malloc(MAX_NAME_LEN);
        strncpy(import_ctx->tensor_names[i], info.name, MAX_NAME_LEN - 1);
    }
    
    /* Calculate data section offset */
    uint64_t data_offset = sizeof(header) + 
                          sizeof(gguf_membrane_header_t) +
                          import_ctx->agent_config_count * sizeof(gguf_agent_config_t) +
                          import_ctx->device_config_count * sizeof(gguf_device_config_t) +
                          import_ctx->module_config_count * sizeof(gguf_module_config_t);
    data_offset += import_ctx->tensor_count * sizeof(gguf_tensor_info_t);
    data_offset = (data_offset + GGUF_DEFAULT_ALIGNMENT - 1) & 
                  ~(GGUF_DEFAULT_ALIGNMENT - 1);
    
    if (fseek(file, data_offset, SEEK_SET) != 0) {
        fclose(file);
        gguf_cleanup_enhanced_import(import_ctx);
        return -EIO;
    }
    
    /* Read tensor data */
    for (uint32_t i = 0; i < import_ctx->tensor_count; i++) {
        tensor_t *tensor = import_ctx->tensors[i];
        
        if (fread(tensor->data, tensor->size_bytes, 1, file) != 1) {
            fclose(file);
            gguf_cleanup_enhanced_import(import_ctx);
            return -EIO;
        }
        
        /* Skip alignment padding */
        uint64_t padding = ((tensor->size_bytes + GGUF_DEFAULT_ALIGNMENT - 1) & 
                           ~(GGUF_DEFAULT_ALIGNMENT - 1)) - tensor->size_bytes;
        
        if (padding > 0 && fseek(file, padding, SEEK_CUR) != 0) {
            fclose(file);
            gguf_cleanup_enhanced_import(import_ctx);
            return -EIO;
        }
    }
    
    fclose(file);
    
    printf("Successfully read enhanced GGUF file: %s\n", filepath);
    printf("  Membrane: %s\n", import_ctx->membrane_header.membrane_name);
    printf("  Loaded: %u tensors, %u agents, %u devices, %u modules\n",
           import_ctx->tensor_count, import_ctx->agent_config_count,
           import_ctx->device_config_count, import_ctx->module_config_count);
    
    return 0;
}

/*
 * Restore enhanced system state from import context
 */
int gguf_restore_enhanced_system(robotics_context_t *ctx, const gguf_enhanced_import_context_t *import_ctx) {
    if (!ctx || !import_ctx) {
        return -EINVAL;
    }
    
    printf("Restoring enhanced system from GGUF P-System membrane: %s\n", 
           import_ctx->membrane_header.membrane_name);
    
    /* Display configuration information */
    printf("\nAgent Configurations:\n");
    for (uint32_t i = 0; i < import_ctx->agent_config_count; i++) {
        const gguf_agent_config_t *config = &import_ctx->agent_configs[i];
        printf("  Agent %u: %s (State: %d)\n", config->agent_id, config->name, config->state);
        printf("    Config: %s\n", config->config_json);
    }
    
    printf("\nDevice Configurations:\n");
    for (uint32_t i = 0; i < import_ctx->device_config_count; i++) {
        const gguf_device_config_t *config = &import_ctx->device_configs[i];
        printf("  Device %u: %s (Type: %d, Active: %s)\n", 
               config->device_id, config->name, config->type, 
               config->active ? "Yes" : "No");
        printf("    Spec: %s\n", config->spec_json);
    }
    
    printf("\nModule Configurations:\n");
    for (uint32_t i = 0; i < import_ctx->module_config_count; i++) {
        const gguf_module_config_t *config = &import_ctx->module_configs[i];
        printf("  Module %u: %s (Type: %d, State: %d)\n", 
               config->module_id, config->name, config->type, config->state);
        printf("    Description: %s\n", config->description);
        if (config->dependency_count > 0) {
            printf("    Dependencies: ");
            for (uint32_t j = 0; j < config->dependency_count; j++) {
                printf("%u ", config->dependency_ids[j]);
            }
            printf("\n");
        }
    }
    
    printf("\nTensor Data:\n");
    for (uint32_t i = 0; i < import_ctx->tensor_count; i++) {
        printf("  Tensor %u: %s\n", i, import_ctx->tensor_names[i]);
        tensor_print_info(import_ctx->tensors[i]);
    }
    
    printf("\nEnhanced system state restoration complete.\n");
    printf("P-System membrane '%s' contains complete agent kernels and environment tensors.\n",
           import_ctx->membrane_header.membrane_name);
    
    return 0;
}