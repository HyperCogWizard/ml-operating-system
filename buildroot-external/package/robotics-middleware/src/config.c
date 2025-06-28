/*
 * Configuration Management for Robotics Middleware
 * 
 * Handles loading and parsing of configuration files
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "robotics_middleware.h"

/*
 * Load configuration from file
 */
int robotics_load_config(robotics_context_t *ctx, const char *config_path) {
    if (!ctx || !config_path) {
        return -EINVAL;
    }
    
    FILE *file = fopen(config_path, "r");
    if (!file) {
        /* If config file doesn't exist, use defaults */
        printf("Config file not found, using defaults: %s\n", config_path);
        return 0;
    }
    
    char line[256];
    
    printf("Loading configuration from: %s\n", config_path);
    
    while (fgets(line, sizeof(line), file)) {
        /* Remove newline */
        line[strcspn(line, "\n")] = 0;
        
        /* Skip empty lines and comments */
        if (line[0] == '\0' || line[0] == '#') {
            continue;
        }
        
        /* Parse key=value pairs */
        char *key = strtok(line, "=");
        char *value = strtok(NULL, "=");
        
        if (!key || !value) {
            continue;
        }
        
        /* Trim whitespace */
        while (*key == ' ' || *key == '\t') key++;
        while (*value == ' ' || *value == '\t') value++;
        
        /* Process configuration options */
        if (strcmp(key, "tensor_memory_limit") == 0) {
            uint64_t limit = strtoull(value, NULL, 10);
            if (limit > 0) {
                ctx->tensor_sys.total_memory = limit;
                printf("Set tensor memory limit: %lu bytes\n", limit);
            }
        } else if (strcmp(key, "max_devices") == 0) {
            /* This would require dynamic allocation, so just log for now */
            printf("Max devices setting: %s (using compile-time limit)\n", value);
        } else if (strcmp(key, "max_agents") == 0) {
            /* This would require dynamic allocation, so just log for now */
            printf("Max agents setting: %s (using compile-time limit)\n", value);
        } else if (strcmp(key, "gguf_version") == 0) {
            strncpy(ctx->gguf_ctx.version, value, sizeof(ctx->gguf_ctx.version) - 1);
            printf("GGUF version: %s\n", value);
        } else {
            printf("Unknown config option: %s = %s\n", key, value);
        }
    }
    
    fclose(file);
    strncpy(ctx->config_path, config_path, sizeof(ctx->config_path) - 1);
    
    printf("Configuration loaded successfully\n");
    return 0;
}