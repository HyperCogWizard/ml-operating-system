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

/* Sensor modalities */
typedef enum {
    MODALITY_VISUAL = 0,
    MODALITY_AUDITORY,
    MODALITY_HAPTIC,
    MODALITY_THERMAL,
    MODALITY_PROXIMITY,
    MODALITY_INERTIAL,
    MODALITY_CHEMICAL,
    MODALITY_ELECTROMAGNETIC,
    MODALITY_CUSTOM
} sensor_modality_t;

/* Actuator degrees of freedom types */
typedef enum {
    DOF_TYPE_ROTATIONAL = 0,
    DOF_TYPE_LINEAR,
    DOF_TYPE_PLANAR,
    DOF_TYPE_SPHERICAL,
    DOF_TYPE_CUSTOM
} dof_type_t;

/* Channel types for multi-channel sensors */
typedef enum {
    CHANNEL_TYPE_RGB = 0,
    CHANNEL_TYPE_RGBA,
    CHANNEL_TYPE_DEPTH,
    CHANNEL_TYPE_MONO,
    CHANNEL_TYPE_STEREO,
    CHANNEL_TYPE_MULTICHANNEL,
    CHANNEL_TYPE_CUSTOM
} channel_type_t;

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
typedef struct workbench_module workbench_module_t;
typedef struct module_registry module_registry_t;

/* Enhanced tensor specification with robotics semantics */
struct tensor_spec {
    uint32_t dimensions;
    uint32_t shape[8];      /* Max 8 dimensions */
    tensor_dtype_t dtype;
    bool requires_grad;
    char metadata[MAX_CONFIG_LEN];
    
    /* Robotics-specific extensions */
    union {
        struct {
            sensor_modality_t modality;
            channel_type_t channel_type;
            uint32_t num_channels;
            uint32_t width;
            uint32_t height;
            float sampling_rate;    /* For temporal sensors */
        } sensor;
        
        struct {
            uint32_t num_dof;
            dof_type_t dof_types[8];    /* Type for each DOF */
            float min_limits[8];        /* Min values for each DOF */
            float max_limits[8];        /* Max values for each DOF */
            float max_velocities[8];    /* Max velocities for each DOF */
        } actuator;
        
        struct {
            uint32_t cognitive_layers;
            uint32_t memory_capacity;
            uint32_t processing_units;
        } processor;
    } semantics;
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

/* Workbench module types */
typedef enum {
    MODULE_TYPE_SENSOR = 0,
    MODULE_TYPE_ACTUATOR,
    MODULE_TYPE_PROCESSOR,
    MODULE_TYPE_COMMUNICATION,
    MODULE_TYPE_COMPOSITE
} module_type_t;

/* Workbench module states */
typedef enum {
    MODULE_STATE_UNINITIALIZED = 0,
    MODULE_STATE_INITIALIZING,
    MODULE_STATE_READY,
    MODULE_STATE_ACTIVE,
    MODULE_STATE_ERROR,
    MODULE_STATE_SHUTTING_DOWN
} module_state_t;

/* Module capabilities */
typedef struct {
    bool supports_streaming;
    bool supports_prediction;
    bool supports_learning;
    bool supports_configuration;
    uint32_t max_input_tensors;
    uint32_t max_output_tensors;
} module_capabilities_t;

/* Workbench module */
struct workbench_module {
    uint32_t id;
    char name[MAX_NAME_LEN];
    char description[MAX_CONFIG_LEN];
    module_type_t type;
    module_state_t state;
    module_capabilities_t capabilities;
    
    /* Device/tensor associations */
    uint32_t device_count;
    device_node_t *devices[8];  /* Max 8 devices per module */
    
    /* Input/output tensors */
    uint32_t input_count;
    uint32_t output_count;
    tensor_t *inputs[8];
    tensor_t *outputs[8];
    
    /* Module-specific data */
    void *module_data;
    
    /* Function pointers for module operations */
    int (*init)(workbench_module_t *module, const char *config);
    int (*process)(workbench_module_t *module);
    int (*configure)(workbench_module_t *module, const char *config);
    void (*cleanup)(workbench_module_t *module);
    
    /* Dependencies */
    uint32_t dependency_count;
    workbench_module_t *dependencies[8];
    
    uint64_t last_update;
    pthread_mutex_t mutex;
};

/* Module registry */
struct module_registry {
    workbench_module_t *modules[MAX_DEVICES];  /* Reuse MAX_DEVICES for max modules */
    uint32_t module_count;
    pthread_rwlock_t lock;
};

/* Enhanced hypergraph structure */
struct hypergraph {
    device_node_t *devices[MAX_DEVICES];
    agent_t *agents[MAX_AGENTS];
    workbench_module_t *modules[MAX_DEVICES];  /* Modules in the workbench */
    uint32_t device_count;
    uint32_t agent_count;
    uint32_t module_count;
    pthread_rwlock_t lock;
    
    /* Module composition graph */
    bool module_connections[MAX_DEVICES][MAX_DEVICES];  /* Adjacency matrix for module connections */
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
    module_registry_t *module_registry;
    tensor_system_t tensor_sys;
    gguf_context_t gguf_ctx;
    uint32_t next_device_id;
    uint32_t next_agent_id;
    uint32_t next_module_id;
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

/* Enhanced GGUF integration with P-System support */
int robotics_export_enhanced_gguf(const char *filepath, const char *membrane_name);
int robotics_import_enhanced_gguf(const char *filepath);

/* Tensor operations */
tensor_t* tensor_create(const tensor_spec_t *spec);
void tensor_destroy(tensor_t *tensor);
int tensor_copy(tensor_t *dst, const tensor_t *src);
uint64_t tensor_size_bytes(const tensor_spec_t *spec);

/* Hypergraph operations */
int hypergraph_add_device(hypergraph_t *graph, device_node_t *device);
int hypergraph_add_agent(hypergraph_t *graph, agent_t *agent);
int hypergraph_add_module(hypergraph_t *graph, workbench_module_t *module);
int hypergraph_remove_device(hypergraph_t *graph, uint32_t device_id);
int hypergraph_remove_agent(hypergraph_t *graph, uint32_t agent_id);
int hypergraph_remove_module(hypergraph_t *graph, uint32_t module_id);
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

/* Workbench module operations */
workbench_module_t* workbench_create_module(const char *name, module_type_t type, 
                                           const char *description);
int workbench_register_module(workbench_module_t *module);
int workbench_connect_modules(workbench_module_t *producer, workbench_module_t *consumer);
int workbench_disconnect_modules(workbench_module_t *producer, workbench_module_t *consumer);
int workbench_execute_module(workbench_module_t *module);
int workbench_execute_pipeline(workbench_module_t **modules, uint32_t count);
void workbench_destroy_module(workbench_module_t *module);

/* Enhanced device creation with semantic specifications */
device_node_t* robotics_create_camera_device(const char *name, uint32_t width, uint32_t height,
                                            channel_type_t channel_type, float fps);
device_node_t* robotics_create_robot_arm_device(const char *name, uint32_t num_dof,
                                               const dof_type_t *dof_types,
                                               const float *min_limits, const float *max_limits);
device_node_t* robotics_create_imu_device(const char *name, float sampling_rate);
device_node_t* robotics_create_audio_device(const char *name, uint32_t channels, float sampling_rate);

/* Pre-built workbench modules */
workbench_module_t* workbench_create_vision_module(const char *name);
workbench_module_t* workbench_create_control_module(const char *name);
workbench_module_t* workbench_create_navigation_module(const char *name);
workbench_module_t* workbench_create_manipulation_module(const char *name);

/* Module registry operations */
int module_registry_init(module_registry_t *registry);
void module_registry_cleanup(module_registry_t *registry);
workbench_module_t* module_registry_find(module_registry_t *registry, const char *name);
int module_registry_add(module_registry_t *registry, workbench_module_t *module);
int module_registry_remove(module_registry_t *registry, uint32_t module_id);

/* Get access to global module registry for GGUF export */
module_registry_t* workbench_get_module_registry(void);

#endif /* ROBOTICS_MIDDLEWARE_H */