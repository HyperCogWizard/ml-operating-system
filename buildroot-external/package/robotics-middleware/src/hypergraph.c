/*
 * Hypergraph Implementation for Robotics Middleware
 * 
 * Manages the hypergraph structure for devices and agents
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "robotics_middleware.h"

/*
 * Add device to hypergraph
 */
int hypergraph_add_device(hypergraph_t *graph, device_node_t *device) {
    if (!graph || !device) {
        return -EINVAL;
    }
    
    pthread_rwlock_wrlock(&graph->lock);
    
    if (graph->device_count >= MAX_DEVICES) {
        pthread_rwlock_unlock(&graph->lock);
        return -ENOSPC;
    }
    
    graph->devices[graph->device_count] = device;
    graph->device_count++;
    
    pthread_rwlock_unlock(&graph->lock);
    
    printf("Added device to hypergraph: %s (total: %u)\n", 
           device->name, graph->device_count);
    return 0;
}

/*
 * Add agent to hypergraph
 */
int hypergraph_add_agent(hypergraph_t *graph, agent_t *agent) {
    if (!graph || !agent) {
        return -EINVAL;
    }
    
    pthread_rwlock_wrlock(&graph->lock);
    
    if (graph->agent_count >= MAX_AGENTS) {
        pthread_rwlock_unlock(&graph->lock);
        return -ENOSPC;
    }
    
    graph->agents[graph->agent_count] = agent;
    graph->agent_count++;
    
    pthread_rwlock_unlock(&graph->lock);
    
    printf("Added agent to hypergraph: %s (total: %u)\n", 
           agent->name, graph->agent_count);
    return 0;
}

/*
 * Remove device from hypergraph
 */
int hypergraph_remove_device(hypergraph_t *graph, uint32_t device_id) {
    if (!graph) {
        return -EINVAL;
    }
    
    pthread_rwlock_wrlock(&graph->lock);
    
    for (uint32_t i = 0; i < graph->device_count; i++) {
        if (graph->devices[i] && graph->devices[i]->id == device_id) {
            /* Free the device */
            if (graph->devices[i]->tensor) {
                tensor_destroy(graph->devices[i]->tensor);
            }
            free(graph->devices[i]);
            
            /* Shift remaining devices */
            for (uint32_t j = i; j < graph->device_count - 1; j++) {
                graph->devices[j] = graph->devices[j + 1];
            }
            graph->devices[graph->device_count - 1] = NULL;
            graph->device_count--;
            
            pthread_rwlock_unlock(&graph->lock);
            
            printf("Removed device %u from hypergraph (remaining: %u)\n", 
                   device_id, graph->device_count);
            return 0;
        }
    }
    
    pthread_rwlock_unlock(&graph->lock);
    return -ENOENT;
}

/*
 * Remove agent from hypergraph
 */
int hypergraph_remove_agent(hypergraph_t *graph, uint32_t agent_id) {
    if (!graph) {
        return -EINVAL;
    }
    
    pthread_rwlock_wrlock(&graph->lock);
    
    for (uint32_t i = 0; i < graph->agent_count; i++) {
        if (graph->agents[i] && graph->agents[i]->id == agent_id) {
            /* Free the agent */
            if (graph->agents[i]->cognitive_tensor) {
                tensor_destroy(graph->agents[i]->cognitive_tensor);
            }
            free(graph->agents[i]);
            
            /* Shift remaining agents */
            for (uint32_t j = i; j < graph->agent_count - 1; j++) {
                graph->agents[j] = graph->agents[j + 1];
            }
            graph->agents[graph->agent_count - 1] = NULL;
            graph->agent_count--;
            
            pthread_rwlock_unlock(&graph->lock);
            
            printf("Removed agent %u from hypergraph (remaining: %u)\n", 
                   agent_id, graph->agent_count);
            return 0;
        }
    }
    
    pthread_rwlock_unlock(&graph->lock);
    return -ENOENT;
}

/*
 * Add module to hypergraph
 */
int hypergraph_add_module(hypergraph_t *graph, workbench_module_t *module) {
    if (!graph || !module) {
        return -EINVAL;
    }
    
    pthread_rwlock_wrlock(&graph->lock);
    
    if (graph->module_count >= MAX_DEVICES) {
        pthread_rwlock_unlock(&graph->lock);
        return -ENOSPC;
    }
    
    graph->modules[graph->module_count] = module;
    graph->module_count++;
    
    pthread_rwlock_unlock(&graph->lock);
    
    printf("Added module to hypergraph: %s (total: %u)\n", 
           module->name, graph->module_count);
    return 0;
}

/*
 * Remove module from hypergraph
 */
int hypergraph_remove_module(hypergraph_t *graph, uint32_t module_id) {
    if (!graph) {
        return -EINVAL;
    }
    
    pthread_rwlock_wrlock(&graph->lock);
    
    for (uint32_t i = 0; i < graph->module_count; i++) {
        if (graph->modules[i] && graph->modules[i]->id == module_id) {
            /* Clear connections for this module */
            for (uint32_t j = 0; j < MAX_DEVICES; j++) {
                graph->module_connections[i][j] = false;
                graph->module_connections[j][i] = false;
            }
            
            /* Destroy the module */
            workbench_destroy_module(graph->modules[i]);
            
            /* Shift remaining modules */
            for (uint32_t j = i; j < graph->module_count - 1; j++) {
                graph->modules[j] = graph->modules[j + 1];
            }
            graph->modules[graph->module_count - 1] = NULL;
            graph->module_count--;
            
            pthread_rwlock_unlock(&graph->lock);
            
            printf("Removed module %u from hypergraph (remaining: %u)\n", 
                   module_id, graph->module_count);
            return 0;
        }
    }
    
    pthread_rwlock_unlock(&graph->lock);
    return -ENOENT;
}
void hypergraph_cleanup(hypergraph_t *graph) {
    if (!graph) {
        return;
    }
    
    pthread_rwlock_wrlock(&graph->lock);
    
    /* Free all devices */
    for (uint32_t i = 0; i < graph->device_count; i++) {
        if (graph->devices[i]) {
            if (graph->devices[i]->tensor) {
                tensor_destroy(graph->devices[i]->tensor);
            }
            free(graph->devices[i]);
            graph->devices[i] = NULL;
        }
    }
    graph->device_count = 0;
    
    /* Free all agents */
    for (uint32_t i = 0; i < graph->agent_count; i++) {
        if (graph->agents[i]) {
            if (graph->agents[i]->cognitive_tensor) {
                tensor_destroy(graph->agents[i]->cognitive_tensor);
            }
            free(graph->agents[i]);
            graph->agents[i] = NULL;
        }
    }
    graph->agent_count = 0;
    
    /* Free all modules */
    for (uint32_t i = 0; i < graph->module_count; i++) {
        if (graph->modules[i]) {
            workbench_destroy_module(graph->modules[i]);
            graph->modules[i] = NULL;
        }
    }
    graph->module_count = 0;
    
    /* Clear module connections */
    memset(graph->module_connections, 0, sizeof(graph->module_connections));
    
    pthread_rwlock_unlock(&graph->lock);
    pthread_rwlock_destroy(&graph->lock);
    
    printf("Hypergraph cleaned up\n");
}