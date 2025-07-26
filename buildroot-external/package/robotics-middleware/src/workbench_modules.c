/*
 * Workbench Modules Implementation
 * 
 * Implements the composable workbench module system for robotics middleware
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>

#include "robotics_middleware.h"
#include "tensor_abstraction.h"

/* Global module registry */
static module_registry_t g_module_registry = {0};
static bool g_registry_initialized = false;
static pthread_mutex_t g_registry_mutex = PTHREAD_MUTEX_INITIALIZER;

/* External references to global robotics context */
extern robotics_context_t g_robotics_ctx;
extern bool g_initialized;
extern pthread_mutex_t g_ctx_mutex;

/*
 * Initialize module registry
 */
int module_registry_init(module_registry_t *registry) {
    if (!registry) {
        return -EINVAL;
    }
    
    memset(registry, 0, sizeof(module_registry_t));
    
    if (pthread_rwlock_init(&registry->lock, NULL) != 0) {
        return -EINVAL;
    }
    
    return 0;
}

/*
 * Cleanup module registry
 */
void module_registry_cleanup(module_registry_t *registry) {
    if (!registry) {
        return;
    }
    
    pthread_rwlock_wrlock(&registry->lock);
    
    /* Cleanup all modules */
    for (uint32_t i = 0; i < registry->module_count; i++) {
        if (registry->modules[i]) {
            workbench_destroy_module(registry->modules[i]);
            registry->modules[i] = NULL;
        }
    }
    registry->module_count = 0;
    
    pthread_rwlock_unlock(&registry->lock);
    pthread_rwlock_destroy(&registry->lock);
}

/*
 * Find module by name
 */
workbench_module_t* module_registry_find(module_registry_t *registry, const char *name) {
    if (!registry || !name) {
        return NULL;
    }
    
    pthread_rwlock_rdlock(&registry->lock);
    
    for (uint32_t i = 0; i < registry->module_count; i++) {
        if (registry->modules[i] && strcmp(registry->modules[i]->name, name) == 0) {
            pthread_rwlock_unlock(&registry->lock);
            return registry->modules[i];
        }
    }
    
    pthread_rwlock_unlock(&registry->lock);
    return NULL;
}

/*
 * Add module to registry
 */
int module_registry_add(module_registry_t *registry, workbench_module_t *module) {
    if (!registry || !module) {
        return -EINVAL;
    }
    
    pthread_rwlock_wrlock(&registry->lock);
    
    if (registry->module_count >= MAX_DEVICES) {
        pthread_rwlock_unlock(&registry->lock);
        return -ENOSPC;
    }
    
    registry->modules[registry->module_count] = module;
    registry->module_count++;
    
    pthread_rwlock_unlock(&registry->lock);
    
    printf("Added module to registry: %s (total: %u)\n", module->name, registry->module_count);
    return 0;
}

/*
 * Remove module from registry
 */
int module_registry_remove(module_registry_t *registry, uint32_t module_id) {
    if (!registry) {
        return -EINVAL;
    }
    
    pthread_rwlock_wrlock(&registry->lock);
    
    for (uint32_t i = 0; i < registry->module_count; i++) {
        if (registry->modules[i] && registry->modules[i]->id == module_id) {
            workbench_destroy_module(registry->modules[i]);
            
            /* Shift remaining modules */
            for (uint32_t j = i; j < registry->module_count - 1; j++) {
                registry->modules[j] = registry->modules[j + 1];
            }
            registry->modules[registry->module_count - 1] = NULL;
            registry->module_count--;
            
            pthread_rwlock_unlock(&registry->lock);
            return 0;
        }
    }
    
    pthread_rwlock_unlock(&registry->lock);
    return -ENOENT;
}

/*
 * Create a new workbench module
 */
workbench_module_t* workbench_create_module(const char *name, module_type_t type, 
                                           const char *description) {
    if (!name || !g_initialized) {
        return NULL;
    }
    
    workbench_module_t *module = calloc(1, sizeof(workbench_module_t));
    if (!module) {
        return NULL;
    }
    
    strncpy(module->name, name, sizeof(module->name) - 1);
    if (description) {
        strncpy(module->description, description, sizeof(module->description) - 1);
    }
    
    module->id = g_robotics_ctx.next_module_id++;
    module->type = type;
    module->state = MODULE_STATE_UNINITIALIZED;
    
    /* Initialize capabilities based on type */
    switch (type) {
        case MODULE_TYPE_SENSOR:
            module->capabilities.supports_streaming = true;
            module->capabilities.max_output_tensors = 4;
            break;
        case MODULE_TYPE_ACTUATOR:
            module->capabilities.supports_configuration = true;
            module->capabilities.max_input_tensors = 4;
            break;
        case MODULE_TYPE_PROCESSOR:
            module->capabilities.supports_prediction = true;
            module->capabilities.supports_learning = true;
            module->capabilities.max_input_tensors = 8;
            module->capabilities.max_output_tensors = 8;
            break;
        case MODULE_TYPE_COMMUNICATION:
            module->capabilities.supports_streaming = true;
            module->capabilities.max_input_tensors = 2;
            module->capabilities.max_output_tensors = 2;
            break;
        case MODULE_TYPE_COMPOSITE:
            module->capabilities.supports_streaming = true;
            module->capabilities.supports_prediction = true;
            module->capabilities.supports_configuration = true;
            module->capabilities.max_input_tensors = 8;
            module->capabilities.max_output_tensors = 8;
            break;
    }
    
    if (pthread_mutex_init(&module->mutex, NULL) != 0) {
        free(module);
        return NULL;
    }
    
    module->state = MODULE_STATE_READY;
    
    printf("Created workbench module: %s (ID: %u, Type: %d)\n", name, module->id, type);
    return module;
}

/*
 * Register a module in the global registry
 */
int workbench_register_module(workbench_module_t *module) {
    if (!module || !g_initialized) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&g_registry_mutex);
    
    if (!g_registry_initialized) {
        if (module_registry_init(&g_module_registry) < 0) {
            pthread_mutex_unlock(&g_registry_mutex);
            return -EINVAL;
        }
        g_registry_initialized = true;
    }
    
    int ret = module_registry_add(&g_module_registry, module);
    pthread_mutex_unlock(&g_registry_mutex);
    
    return ret;
}

/*
 * Connect two modules in a producer-consumer relationship
 */
int workbench_connect_modules(workbench_module_t *producer, workbench_module_t *consumer) {
    if (!producer || !consumer || !g_initialized) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&producer->mutex);
    pthread_mutex_lock(&consumer->mutex);
    
    /* Add consumer as dependency of producer */
    if (producer->dependency_count >= 8) {
        pthread_mutex_unlock(&consumer->mutex);
        pthread_mutex_unlock(&producer->mutex);
        return -ENOSPC;
    }
    
    producer->dependencies[producer->dependency_count] = consumer;
    producer->dependency_count++;
    
    /* Update hypergraph connection matrix */
    if (g_robotics_ctx.hypergraph && 
        producer->id < MAX_DEVICES && consumer->id < MAX_DEVICES) {
        pthread_rwlock_wrlock(&g_robotics_ctx.hypergraph->lock);
        g_robotics_ctx.hypergraph->module_connections[producer->id][consumer->id] = true;
        pthread_rwlock_unlock(&g_robotics_ctx.hypergraph->lock);
    }
    
    pthread_mutex_unlock(&consumer->mutex);
    pthread_mutex_unlock(&producer->mutex);
    
    printf("Connected modules: %s -> %s\n", producer->name, consumer->name);
    return 0;
}

/*
 * Disconnect two modules
 */
int workbench_disconnect_modules(workbench_module_t *producer, workbench_module_t *consumer) {
    if (!producer || !consumer || !g_initialized) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&producer->mutex);
    
    /* Remove consumer from producer's dependencies */
    for (uint32_t i = 0; i < producer->dependency_count; i++) {
        if (producer->dependencies[i] == consumer) {
            /* Shift remaining dependencies */
            for (uint32_t j = i; j < producer->dependency_count - 1; j++) {
                producer->dependencies[j] = producer->dependencies[j + 1];
            }
            producer->dependencies[producer->dependency_count - 1] = NULL;
            producer->dependency_count--;
            break;
        }
    }
    
    /* Update hypergraph connection matrix */
    if (g_robotics_ctx.hypergraph && 
        producer->id < MAX_DEVICES && consumer->id < MAX_DEVICES) {
        pthread_rwlock_wrlock(&g_robotics_ctx.hypergraph->lock);
        g_robotics_ctx.hypergraph->module_connections[producer->id][consumer->id] = false;
        pthread_rwlock_unlock(&g_robotics_ctx.hypergraph->lock);
    }
    
    pthread_mutex_unlock(&producer->mutex);
    
    printf("Disconnected modules: %s -> %s\n", producer->name, consumer->name);
    return 0;
}

/*
 * Execute a single module
 */
int workbench_execute_module(workbench_module_t *module) {
    if (!module) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&module->mutex);
    
    if (module->state != MODULE_STATE_READY && module->state != MODULE_STATE_ACTIVE) {
        pthread_mutex_unlock(&module->mutex);
        return -ENOTCONN;  /* Use ENOTCONN instead of ENOTREADY */
    }
    
    module->state = MODULE_STATE_ACTIVE;
    
    int ret = 0;
    if (module->process) {
        ret = module->process(module);
    }
    
    module->last_update = time(NULL);
    
    if (ret == 0) {
        module->state = MODULE_STATE_READY;
    } else {
        module->state = MODULE_STATE_ERROR;
    }
    
    pthread_mutex_unlock(&module->mutex);
    
    return ret;
}

/*
 * Execute a pipeline of modules in sequence
 */
int workbench_execute_pipeline(workbench_module_t **modules, uint32_t count) {
    if (!modules || count == 0) {
        return -EINVAL;
    }
    
    for (uint32_t i = 0; i < count; i++) {
        int ret = workbench_execute_module(modules[i]);
        if (ret < 0) {
            printf("Pipeline execution failed at module %u (%s): %d\n", 
                   i, modules[i] ? modules[i]->name : "NULL", ret);
            return ret;
        }
    }
    
    printf("Executed pipeline with %u modules\n", count);
    return 0;
}

/*
 * Destroy a workbench module
 */
void workbench_destroy_module(workbench_module_t *module) {
    if (!module) {
        return;
    }
    
    pthread_mutex_lock(&module->mutex);
    
    module->state = MODULE_STATE_SHUTTING_DOWN;
    
    /* Cleanup module-specific data */
    if (module->cleanup) {
        module->cleanup(module);
    }
    
    /* Cleanup tensors */
    for (uint32_t i = 0; i < module->input_count; i++) {
        if (module->inputs[i]) {
            tensor_destroy(module->inputs[i]);
        }
    }
    for (uint32_t i = 0; i < module->output_count; i++) {
        if (module->outputs[i]) {
            tensor_destroy(module->outputs[i]);
        }
    }
    
    pthread_mutex_unlock(&module->mutex);
    pthread_mutex_destroy(&module->mutex);
    
    printf("Destroyed workbench module: %s\n", module->name);
    free(module);
}

/*
 * Enhanced device creation functions with semantic specifications
 */

device_node_t* robotics_create_camera_device(const char *name, uint32_t width, uint32_t height,
                                            channel_type_t channel_type, float fps) {
    if (!name || width == 0 || height == 0) {
        return NULL;
    }
    
    tensor_spec_t spec = {0};
    
    /* Determine channels based on type */
    uint32_t channels = 1;
    switch (channel_type) {
        case CHANNEL_TYPE_RGB: channels = 3; break;
        case CHANNEL_TYPE_RGBA: channels = 4; break;
        case CHANNEL_TYPE_DEPTH: channels = 1; break;
        case CHANNEL_TYPE_MONO: channels = 1; break;
        default: channels = 3; break;
    }
    
    spec.dimensions = 3;
    spec.shape[0] = height;
    spec.shape[1] = width;
    spec.shape[2] = channels;
    spec.dtype = TENSOR_UINT8;
    spec.requires_grad = false;
    
    /* Set sensor semantics */
    spec.semantics.sensor.modality = MODALITY_VISUAL;
    spec.semantics.sensor.channel_type = channel_type;
    spec.semantics.sensor.num_channels = channels;
    spec.semantics.sensor.width = width;
    spec.semantics.sensor.height = height;
    spec.semantics.sensor.sampling_rate = fps;
    
    snprintf(spec.metadata, sizeof(spec.metadata), "camera_%ux%u_%uch_%.1ffps", 
             width, height, channels, fps);
    
    return robotics_create_device(name, DEVICE_TYPE_SENSOR, &spec);
}

device_node_t* robotics_create_robot_arm_device(const char *name, uint32_t num_dof,
                                               const dof_type_t *dof_types,
                                               const float *min_limits, const float *max_limits) {
    if (!name || num_dof == 0 || num_dof > 8) {
        return NULL;
    }
    
    tensor_spec_t spec = {0};
    
    spec.dimensions = 1;
    spec.shape[0] = num_dof;
    spec.dtype = TENSOR_FLOAT32;
    spec.requires_grad = false;
    
    /* Set actuator semantics */
    spec.semantics.actuator.num_dof = num_dof;
    for (uint32_t i = 0; i < num_dof; i++) {
        spec.semantics.actuator.dof_types[i] = dof_types ? dof_types[i] : DOF_TYPE_ROTATIONAL;
        spec.semantics.actuator.min_limits[i] = min_limits ? min_limits[i] : -3.14159f;
        spec.semantics.actuator.max_limits[i] = max_limits ? max_limits[i] : 3.14159f;
        spec.semantics.actuator.max_velocities[i] = 1.0f;  /* Default 1 rad/s */
    }
    
    snprintf(spec.metadata, sizeof(spec.metadata), "robot_arm_%udof", num_dof);
    
    return robotics_create_device(name, DEVICE_TYPE_ACTUATOR, &spec);
}

device_node_t* robotics_create_imu_device(const char *name, float sampling_rate) {
    if (!name || sampling_rate <= 0) {
        return NULL;
    }
    
    tensor_spec_t spec = {0};
    
    /* IMU typically has 9 values: 3 accel + 3 gyro + 3 mag */
    spec.dimensions = 1;
    spec.shape[0] = 9;
    spec.dtype = TENSOR_FLOAT32;
    spec.requires_grad = false;
    
    /* Set sensor semantics */
    spec.semantics.sensor.modality = MODALITY_INERTIAL;
    spec.semantics.sensor.channel_type = CHANNEL_TYPE_MULTICHANNEL;
    spec.semantics.sensor.num_channels = 9;
    spec.semantics.sensor.sampling_rate = sampling_rate;
    
    snprintf(spec.metadata, sizeof(spec.metadata), "imu_9axis_%.1fHz", sampling_rate);
    
    return robotics_create_device(name, DEVICE_TYPE_SENSOR, &spec);
}

device_node_t* robotics_create_audio_device(const char *name, uint32_t channels, float sampling_rate) {
    if (!name || channels == 0 || sampling_rate <= 0) {
        return NULL;
    }
    
    tensor_spec_t spec = {0};
    
    /* Audio buffer - assume 1024 samples per frame */
    spec.dimensions = 2;
    spec.shape[0] = 1024;
    spec.shape[1] = channels;
    spec.dtype = TENSOR_FLOAT32;
    spec.requires_grad = false;
    
    /* Set sensor semantics */
    spec.semantics.sensor.modality = MODALITY_AUDITORY;
    spec.semantics.sensor.channel_type = (channels == 1) ? CHANNEL_TYPE_MONO : 
                                        (channels == 2) ? CHANNEL_TYPE_STEREO : 
                                        CHANNEL_TYPE_MULTICHANNEL;
    spec.semantics.sensor.num_channels = channels;
    spec.semantics.sensor.sampling_rate = sampling_rate;
    
    snprintf(spec.metadata, sizeof(spec.metadata), "audio_%uch_%.1fHz", channels, sampling_rate);
    
    return robotics_create_device(name, DEVICE_TYPE_SENSOR, &spec);
}

/*
 * Pre-built workbench modules
 */

/* Default processing function for vision modules */
static int vision_module_process(workbench_module_t *module) {
    if (!module) return -EINVAL;
    
    /* Simple pass-through processing for now */
    for (uint32_t i = 0; i < module->input_count && i < module->output_count; i++) {
        if (module->inputs[i] && module->outputs[i]) {
            tensor_copy(module->outputs[i], module->inputs[i]);
        }
    }
    
    return 0;
}

workbench_module_t* workbench_create_vision_module(const char *name) {
    workbench_module_t *module = workbench_create_module(name, MODULE_TYPE_PROCESSOR, 
                                                        "Computer vision processing module");
    if (!module) {
        return NULL;
    }
    
    module->process = vision_module_process;
    
    return module;
}

workbench_module_t* workbench_create_control_module(const char *name) {
    workbench_module_t *module = workbench_create_module(name, MODULE_TYPE_PROCESSOR,
                                                        "Robot control processing module");
    if (!module) {
        return NULL;
    }
    
    module->process = vision_module_process;  /* Use same simple processing for now */
    
    return module;
}

workbench_module_t* workbench_create_navigation_module(const char *name) {
    workbench_module_t *module = workbench_create_module(name, MODULE_TYPE_PROCESSOR,
                                                        "Robot navigation processing module");
    if (!module) {
        return NULL;
    }
    
    module->process = vision_module_process;  /* Use same simple processing for now */
    
    return module;
}

workbench_module_t* workbench_create_manipulation_module(const char *name) {
    workbench_module_t *module = workbench_create_module(name, MODULE_TYPE_PROCESSOR,
                                                        "Robot manipulation processing module");
    if (!module) {
        return NULL;
    }
    
    module->process = vision_module_process;  /* Use same simple processing for now */
    
    return module;
}

/*
 * Get access to the global module registry for GGUF export
 */
module_registry_t* workbench_get_module_registry(void) {
    if (!g_registry_initialized) {
        return NULL;
    }
    return &g_module_registry;
}
