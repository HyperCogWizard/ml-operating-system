/*
 * GGUF Integration Header
 * 
 * Defines GGUF serialization and deserialization functions for complete
 * agent states, device configurations, and P-System membrane support
 */

#ifndef GGUF_INTEGRATION_H
#define GGUF_INTEGRATION_H

#include "robotics_middleware.h"
#include "tensor_abstraction.h"

/* GGUF-specific functions */
int gguf_validate_tensor(const tensor_t *tensor);
int gguf_get_tensor_metadata(const tensor_t *tensor, char *metadata, size_t size);
int gguf_set_tensor_metadata(tensor_t *tensor, const char *metadata);

/* Enhanced GGUF support for complete system state */
typedef struct {
    char name[MAX_NAME_LEN];
    char config_json[MAX_CONFIG_LEN * 4];  /* JSON serialized configuration */
    agent_state_t state;
    uint32_t agent_id;
} gguf_agent_config_t;

typedef struct {
    char name[MAX_NAME_LEN];
    char spec_json[MAX_CONFIG_LEN * 4];   /* JSON serialized device spec */
    device_type_t type;
    uint32_t device_id;
    bool active;
} gguf_device_config_t;

typedef struct {
    char name[MAX_NAME_LEN];
    char description[MAX_CONFIG_LEN];
    module_type_t type;
    module_state_t state;
    uint32_t module_id;
    uint32_t dependency_count;
    uint32_t dependency_ids[8];  /* IDs of dependent modules */
} gguf_module_config_t;

/* P-System membrane structure */
typedef struct {
    char membrane_name[MAX_NAME_LEN];
    uint32_t tensor_count;
    uint32_t agent_count;
    uint32_t device_count;
    uint32_t module_count;
    char metadata[MAX_CONFIG_LEN];
} gguf_membrane_header_t;

/* Enhanced export/import contexts with full system state */
typedef struct {
    uint32_t tensor_count;
    tensor_t **tensors;
    char **tensor_names;
    
    /* Agent configurations */
    uint32_t agent_config_count;
    gguf_agent_config_t *agent_configs;
    
    /* Device configurations */
    uint32_t device_config_count;
    gguf_device_config_t *device_configs;
    
    /* Module configurations */
    uint32_t module_config_count;
    gguf_module_config_t *module_configs;
    
    /* P-System membrane */
    gguf_membrane_header_t membrane_header;
    
    void *metadata;
    uint32_t metadata_size;
} gguf_enhanced_export_context_t;

typedef gguf_enhanced_export_context_t gguf_enhanced_import_context_t;

/* Enhanced GGUF operations for complete system state */
int gguf_prepare_enhanced_export(const robotics_context_t *ctx, gguf_enhanced_export_context_t *export_ctx);
int gguf_write_enhanced_file(const gguf_enhanced_export_context_t *export_ctx, const char *filepath);
int gguf_read_enhanced_file(gguf_enhanced_import_context_t *import_ctx, const char *filepath);
int gguf_restore_enhanced_system(robotics_context_t *ctx, const gguf_enhanced_import_context_t *import_ctx);
void gguf_cleanup_enhanced_export(gguf_enhanced_export_context_t *export_ctx);
void gguf_cleanup_enhanced_import(gguf_enhanced_import_context_t *import_ctx);

/* P-System membrane operations */
int gguf_create_membrane(const char *membrane_name, gguf_enhanced_export_context_t *export_ctx);
int gguf_extract_membrane(const gguf_enhanced_import_context_t *import_ctx, const char *membrane_name);

#endif /* GGUF_INTEGRATION_H */