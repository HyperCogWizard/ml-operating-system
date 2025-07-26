/*
 * Robotics Middleware Abstraction - Header file
 * 
 * Defines the core data structures and API for the robotics middleware
 * with hypergraph-encoded workbench components and tensor abstractions.
 */

#ifndef ROBOTICS_MIDDLEWARE_H
#define ROBOTICS_MIDDLEWARE_H

#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>

/* Maximum string lengths */
#define MAX_NAME_LEN 64
#define MAX_CONFIG_LEN 256
#define MAX_DEVICES 1024
#define MAX_AGENTS 256

/* Device types */
typedef enum {
    DEVICE_TYPE_SENSOR = 0,
    DEVICE_TYPE_ACTUATOR,
    DEVICE_TYPE_PROCESSOR,
    DEVICE_TYPE_COMMUNICATION,
    DEVICE_TYPE_POWER,
    DEVICE_TYPE_CUSTOM
} device_type_t;

/* Agent states */
typedef enum {
    AGENT_STATE_IDLE = 0,
    AGENT_STATE_ACTIVE,
    AGENT_STATE_LEARNING,
    AGENT_STATE_PLANNING,
    AGENT_STATE_EXECUTING,
    AGENT_STATE_ERROR
} agent_state_t;

/* Tensor data types */
typedef enum {
    TENSOR_UINT8 = 0,
    TENSOR_INT8,
    TENSOR_UINT16,
    TENSOR_INT16,
    TENSOR_UINT32,
    TENSOR_INT32,
    TENSOR_FLOAT32,
    TENSOR_FLOAT64
} tensor_dtype_t;

/* Forward declarations */
typedef struct tensor_spec tensor_spec_t;
typedef struct tensor tensor_t;
typedef struct hypergraph hypergraph_t;
typedef struct device_node device_node_t;
typedef struct agent agent_t;
typedef struct tensor_system tensor_system_t;
typedef struct gguf_context gguf_context_t;

/* Tensor specification */
struct tensor_spec {
    uint32_t dimensions;
    uint32_t shape[8];      /* Max 8 dimensions */
    tensor_dtype_t dtype;
    bool requires_grad;
    char metadata[MAX_CONFIG_LEN];
};

/* Tensor structure */
struct tensor {
    tensor_spec_t spec;
    void *data;
    uint64_t size_bytes;
    uint32_t ref_count;
    pthread_mutex_t mutex;
};

/* Device node in hypergraph */
struct device_node {
    uint32_t id;
    char name[MAX_NAME_LEN];
    device_type_t type;
    tensor_spec_t tensor_spec;
    tensor_t *tensor;
    bool active;
    uint64_t last_update;
    void *driver_context;
};

/* Agent configuration */
typedef struct {
    uint32_t cognitive_dimensions;
    uint32_t memory_size;
    uint32_t attention_heads;
    uint32_t embedding_dim;
    char scheme_functions[MAX_CONFIG_LEN];
    char neural_symbolic_config[MAX_CONFIG_LEN];
} agent_config_t;

/* Agent structure */
struct agent {
    uint32_t id;
    char name[MAX_NAME_LEN];
    agent_config_t config;
    agent_state_t state;
    tensor_t *cognitive_tensor;
    bool autonomous;
    uint64_t last_action;
    void *scheme_context;
};

/* Hypergraph structure */
struct hypergraph {
    device_node_t *devices[MAX_DEVICES];
    agent_t *agents[MAX_AGENTS];
    uint32_t device_count;
    uint32_t agent_count;
    pthread_rwlock_t lock;
};

/* Tensor system */
struct tensor_system {
    uint64_t total_memory;
    uint64_t used_memory;
    uint32_t tensor_count;
    pthread_mutex_t alloc_mutex;
};

/* GGUF integration context */
struct gguf_context {
    bool initialized;
    char version[16];
    uint32_t metadata_count;
    void *metadata;
};

/* Main robotics context */
typedef struct {
    hypergraph_t *hypergraph;
    tensor_system_t tensor_sys;
    gguf_context_t gguf_ctx;
    uint32_t next_device_id;
    uint32_t next_agent_id;
    char config_path[MAX_CONFIG_LEN];
} robotics_context_t;

/* System status */
typedef struct {
    bool initialized;
    uint32_t device_count;
    uint32_t agent_count;
    uint64_t tensor_memory_usage;
    uint32_t active_agents;
    uint64_t uptime_seconds;
} robotics_status_t;

/* GGUF export/import contexts */
typedef struct {
    uint32_t tensor_count;
    tensor_t **tensors;
    char **tensor_names;
    void *metadata;
    uint32_t metadata_size;
} gguf_export_context_t;

typedef struct {
    uint32_t tensor_count;
    tensor_t **tensors;
    char **tensor_names;
    void *metadata;
    uint32_t metadata_size;
} gguf_import_context_t;

/* Core API functions */
int robotics_middleware_init(const char *config_path);
void robotics_middleware_cleanup(void);
int robotics_get_status(robotics_status_t *status);

/* Device management */
device_node_t* robotics_create_device(const char *name, device_type_t type, 
                                    const tensor_spec_t *tensor_spec);
int robotics_update_device(device_node_t *device, const void *data, uint64_t size);
int robotics_read_device(device_node_t *device, void *buffer, uint64_t size);

/* Agent management */
agent_t* robotics_create_agent(const char *name, const agent_config_t *config);
int robotics_update_agent_state(agent_t *agent, agent_state_t new_state);
int robotics_agent_execute_scheme(agent_t *agent, const char *scheme_code);

/* GGUF integration */
int robotics_export_gguf(const char *filepath);
int robotics_import_gguf(const char *filepath);

/* Tensor operations */
tensor_t* tensor_create(const tensor_spec_t *spec);
void tensor_destroy(tensor_t *tensor);
int tensor_copy(tensor_t *dst, const tensor_t *src);
uint64_t tensor_size_bytes(const tensor_spec_t *spec);

/* Hypergraph operations */
int hypergraph_add_device(hypergraph_t *graph, device_node_t *device);
int hypergraph_add_agent(hypergraph_t *graph, agent_t *agent);
int hypergraph_remove_device(hypergraph_t *graph, uint32_t device_id);
int hypergraph_remove_agent(hypergraph_t *graph, uint32_t agent_id);
void hypergraph_cleanup(hypergraph_t *graph);

/* Tensor system operations */
int tensor_system_init(tensor_system_t *sys);
void tensor_system_cleanup(tensor_system_t *sys);
uint64_t tensor_system_memory_usage(const tensor_system_t *sys);

/* GGUF operations */
int gguf_integration_init(gguf_context_t *ctx);
void gguf_integration_cleanup(gguf_context_t *ctx);
int gguf_prepare_export(const robotics_context_t *ctx, gguf_export_context_t *export_ctx);
int gguf_write_file(const gguf_export_context_t *export_ctx, const char *filepath);
int gguf_read_file(gguf_import_context_t *import_ctx, const char *filepath);
int gguf_restore_system(robotics_context_t *ctx, const gguf_import_context_t *import_ctx);
void gguf_cleanup_export(gguf_export_context_t *export_ctx);
void gguf_cleanup_import(gguf_import_context_t *import_ctx);

/* Configuration */
int robotics_load_config(robotics_context_t *ctx, const char *config_path);

#endif /* ROBOTICS_MIDDLEWARE_H */