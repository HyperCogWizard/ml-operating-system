/*
 * Robotics Middleware Abstraction - Core Framework
 * 
 * This implements the hypergraph-encoded workbench components for
 * robotics middleware with tensor-based device abstraction.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <pthread.h>
#include <errno.h>

#include "robotics_middleware.h"
#include "tensor_abstraction.h"
#include "gguf_integration.h"

/* Global state for the robotics middleware */
static robotics_context_t g_robotics_ctx = {0};
static bool g_initialized = false;
static pthread_mutex_t g_ctx_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * Initialize the robotics middleware system
 */
int robotics_middleware_init(const char *config_path) {
    pthread_mutex_lock(&g_ctx_mutex);
    
    if (g_initialized) {
        pthread_mutex_unlock(&g_ctx_mutex);
        return 0; /* Already initialized */
    }
    
    memset(&g_robotics_ctx, 0, sizeof(robotics_context_t));
    
    /* Initialize hypergraph structure */
    g_robotics_ctx.hypergraph = calloc(1, sizeof(hypergraph_t));
    if (!g_robotics_ctx.hypergraph) {
        pthread_mutex_unlock(&g_ctx_mutex);
        return -ENOMEM;
    }
    
    /* Initialize tensor system */
    if (tensor_system_init(&g_robotics_ctx.tensor_sys) < 0) {
        free(g_robotics_ctx.hypergraph);
        pthread_mutex_unlock(&g_ctx_mutex);
        return -EINVAL;
    }
    
    /* Initialize GGUF integration */
    if (gguf_integration_init(&g_robotics_ctx.gguf_ctx) < 0) {
        tensor_system_cleanup(&g_robotics_ctx.tensor_sys);
        free(g_robotics_ctx.hypergraph);
        pthread_mutex_unlock(&g_ctx_mutex);
        return -EINVAL;
    }
    
    /* Load configuration */
    if (config_path && robotics_load_config(&g_robotics_ctx, config_path) < 0) {
        gguf_integration_cleanup(&g_robotics_ctx.gguf_ctx);
        tensor_system_cleanup(&g_robotics_ctx.tensor_sys);
        free(g_robotics_ctx.hypergraph);
        pthread_mutex_unlock(&g_ctx_mutex);
        return -EINVAL;
    }
    
    g_initialized = true;
    pthread_mutex_unlock(&g_ctx_mutex);
    
    printf("Robotics Middleware initialized successfully\n");
    return 0;
}

/*
 * Create a new device node in the hypergraph
 */
device_node_t* robotics_create_device(const char *name, device_type_t type, 
                                    const tensor_spec_t *tensor_spec) {
    if (!g_initialized || !name || !tensor_spec) {
        return NULL;
    }
    
    pthread_mutex_lock(&g_ctx_mutex);
    
    device_node_t *device = calloc(1, sizeof(device_node_t));
    if (!device) {
        pthread_mutex_unlock(&g_ctx_mutex);
        return NULL;
    }
    
    strncpy(device->name, name, sizeof(device->name) - 1);
    device->type = type;
    device->tensor_spec = *tensor_spec;
    device->id = g_robotics_ctx.next_device_id++;
    
    /* Create tensor for this device */
    device->tensor = tensor_create(&device->tensor_spec);
    if (!device->tensor) {
        free(device);
        pthread_mutex_unlock(&g_ctx_mutex);
        return NULL;
    }
    
    /* Add to hypergraph */
    hypergraph_add_device(g_robotics_ctx.hypergraph, device);
    
    pthread_mutex_unlock(&g_ctx_mutex);
    
    printf("Created device: %s (ID: %u, Type: %d)\n", name, device->id, type);
    return device;
}

/*
 * Create an agent for autonomous operation
 */
agent_t* robotics_create_agent(const char *name, const agent_config_t *config) {
    if (!g_initialized || !name || !config) {
        return NULL;
    }
    
    pthread_mutex_lock(&g_ctx_mutex);
    
    agent_t *agent = calloc(1, sizeof(agent_t));
    if (!agent) {
        pthread_mutex_unlock(&g_ctx_mutex);
        return NULL;
    }
    
    strncpy(agent->name, name, sizeof(agent->name) - 1);
    agent->config = *config;
    agent->id = g_robotics_ctx.next_agent_id++;
    agent->state = AGENT_STATE_IDLE;
    
    /* Initialize agent's cognitive state tensor */
    tensor_spec_t cognitive_spec = {
        .dimensions = config->cognitive_dimensions,
        .shape = {config->memory_size, config->attention_heads, config->embedding_dim},
        .dtype = TENSOR_FLOAT32
    };
    
    agent->cognitive_tensor = tensor_create(&cognitive_spec);
    if (!agent->cognitive_tensor) {
        free(agent);
        pthread_mutex_unlock(&g_ctx_mutex);
        return NULL;
    }
    
    /* Add to hypergraph */
    hypergraph_add_agent(g_robotics_ctx.hypergraph, agent);
    
    pthread_mutex_unlock(&g_ctx_mutex);
    
    printf("Created agent: %s (ID: %u)\n", name, agent->id);
    return agent;
}

/*
 * Export current system state to GGUF format
 */
int robotics_export_gguf(const char *filepath) {
    if (!g_initialized || !filepath) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&g_ctx_mutex);
    
    gguf_export_context_t export_ctx = {0};
    
    /* Collect all tensors from devices and agents */
    int ret = gguf_prepare_export(&g_robotics_ctx, &export_ctx);
    if (ret < 0) {
        pthread_mutex_unlock(&g_ctx_mutex);
        return ret;
    }
    
    /* Write GGUF file */
    ret = gguf_write_file(&export_ctx, filepath);
    
    /* Cleanup export context */
    gguf_cleanup_export(&export_ctx);
    
    pthread_mutex_unlock(&g_ctx_mutex);
    
    if (ret == 0) {
        printf("Exported system state to GGUF: %s\n", filepath);
    }
    
    return ret;
}

/*
 * Load system state from GGUF format
 */
int robotics_import_gguf(const char *filepath) {
    if (!g_initialized || !filepath) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&g_ctx_mutex);
    
    gguf_import_context_t import_ctx = {0};
    
    /* Read GGUF file */
    int ret = gguf_read_file(&import_ctx, filepath);
    if (ret < 0) {
        pthread_mutex_unlock(&g_ctx_mutex);
        return ret;
    }
    
    /* Restore system state */
    ret = gguf_restore_system(&g_robotics_ctx, &import_ctx);
    
    /* Cleanup import context */
    gguf_cleanup_import(&import_ctx);
    
    pthread_mutex_unlock(&g_ctx_mutex);
    
    if (ret == 0) {
        printf("Imported system state from GGUF: %s\n", filepath);
    }
    
    return ret;
}

/*
 * Cleanup the robotics middleware system
 */
void robotics_middleware_cleanup(void) {
    pthread_mutex_lock(&g_ctx_mutex);
    
    if (!g_initialized) {
        pthread_mutex_unlock(&g_ctx_mutex);
        return;
    }
    
    /* Cleanup GGUF integration */
    gguf_integration_cleanup(&g_robotics_ctx.gguf_ctx);
    
    /* Cleanup tensor system */
    tensor_system_cleanup(&g_robotics_ctx.tensor_sys);
    
    /* Cleanup hypergraph */
    if (g_robotics_ctx.hypergraph) {
        hypergraph_cleanup(g_robotics_ctx.hypergraph);
        free(g_robotics_ctx.hypergraph);
    }
    
    g_initialized = false;
    pthread_mutex_unlock(&g_ctx_mutex);
    
    printf("Robotics Middleware cleaned up\n");
}

/*
 * Get system status
 */
int robotics_get_status(robotics_status_t *status) {
    if (!g_initialized || !status) {
        return -EINVAL;
    }
    
    pthread_mutex_lock(&g_ctx_mutex);
    
    memset(status, 0, sizeof(robotics_status_t));
    status->initialized = true;
    status->device_count = g_robotics_ctx.hypergraph->device_count;
    status->agent_count = g_robotics_ctx.hypergraph->agent_count;
    status->tensor_memory_usage = tensor_system_memory_usage(&g_robotics_ctx.tensor_sys);
    
    pthread_mutex_unlock(&g_ctx_mutex);
    
    return 0;
}