/*
 * Robotics Middleware Main Application
 * 
 * Command-line interface for the robotics engineering workbench
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <getopt.h>
#include <errno.h>

#include "robotics_middleware.h"
#include "tensor_abstraction.h"

static bool g_running = true;

static void signal_handler(int sig) {
    printf("\nReceived signal %d, shutting down...\n", sig);
    g_running = false;
}

static void print_usage(const char *program_name) {
    printf("Usage: %s [OPTIONS]\n", program_name);
    printf("Robotics Engineering Workbench with GGUF Integration\n\n");
    printf("Options:\n");
    printf("  -c, --config PATH     Configuration file path\n");
    printf("  -d, --daemon          Run as daemon\n");
    printf("  -e, --export FILE     Export system state to GGUF file\n");
    printf("  -i, --import FILE     Import system state from GGUF file\n");
    printf("  -s, --status          Show system status\n");
    printf("  -t, --test            Run test scenarios\n");
    printf("  -h, --help            Show this help message\n");
    printf("  -v, --version         Show version information\n");
}

static void print_version(void) {
    printf("Robotics Middleware v1.0.0\n");
    printf("GGUF Integration for Marduk's Robotics Lab\n");
    printf("Built on %s at %s\n", __DATE__, __TIME__);
}

static int run_test_scenarios(void) {
    printf("Running test scenarios...\n");
    
    /* Test 1: Create enhanced sensor devices with semantic specifications */
    device_node_t *camera = robotics_create_camera_device("main_camera", 640, 480, 
                                                          CHANNEL_TYPE_RGB, 30.0f);
    if (!camera) {
        printf("Failed to create camera device\n");
        return -1;
    }
    
    /* Test 2: Create enhanced actuator device */
    dof_type_t arm_dofs[] = {DOF_TYPE_ROTATIONAL, DOF_TYPE_ROTATIONAL, DOF_TYPE_ROTATIONAL,
                            DOF_TYPE_ROTATIONAL, DOF_TYPE_ROTATIONAL, DOF_TYPE_ROTATIONAL};
    float arm_min_limits[] = {-3.14f, -1.57f, -1.57f, -3.14f, -1.57f, -3.14f};
    float arm_max_limits[] = {3.14f, 1.57f, 1.57f, 3.14f, 1.57f, 3.14f};
    
    device_node_t *arm = robotics_create_robot_arm_device("robot_arm", 6, arm_dofs, 
                                                         arm_min_limits, arm_max_limits);
    if (!arm) {
        printf("Failed to create robot arm device\n");
        return -1;
    }
    
    /* Test 3: Create IMU sensor */
    device_node_t *imu = robotics_create_imu_device("main_imu", 100.0f);
    if (!imu) {
        printf("Failed to create IMU device\n");
        return -1;
    }
    
    /* Test 4: Create agent */
    agent_config_t agent_config = {
        .cognitive_dimensions = 3,
        .memory_size = 1024,
        .attention_heads = 8,
        .embedding_dim = 512
    };
    strncpy(agent_config.scheme_functions, "(define control-loop ...)", sizeof(agent_config.scheme_functions) - 1);
    strncpy(agent_config.neural_symbolic_config, "transformer_agent", sizeof(agent_config.neural_symbolic_config) - 1);
    
    agent_t *control_agent = robotics_create_agent("main_controller", &agent_config);
    if (!control_agent) {
        printf("Failed to create control agent\n");
        return -1;
    }
    
    /* Test 5: Create workbench modules */
    workbench_module_t *vision_module = workbench_create_vision_module("vision_processor");
    workbench_module_t *control_module = workbench_create_control_module("arm_controller");
    workbench_module_t *nav_module = workbench_create_navigation_module("navigator");
    
    if (!vision_module || !control_module || !nav_module) {
        printf("Failed to create workbench modules\n");
        return -1;
    }
    
    /* Test 6: Register modules */
    if (workbench_register_module(vision_module) < 0 ||
        workbench_register_module(control_module) < 0 ||
        workbench_register_module(nav_module) < 0) {
        printf("Failed to register modules\n");
        return -1;
    }
    
    /* Test 7: Connect modules in a processing pipeline */
    if (workbench_connect_modules(vision_module, control_module) < 0 ||
        workbench_connect_modules(vision_module, nav_module) < 0) {
        printf("Failed to connect modules\n");
        return -1;
    }
    
    /* Test 8: Update sensor data */
    uint8_t camera_data[640 * 480 * 3];  /* RGB data */
    memset(camera_data, 128, sizeof(camera_data));  /* Gray image */
    
    if (tensor_set_data(camera->tensor, camera_data, sizeof(camera_data)) < 0) {
        printf("Failed to set camera data\n");
        return -1;
    }
    
    /* Test 9: Update actuator commands */
    float arm_positions[6] = {0.0f, 1.57f, -1.57f, 0.0f, 1.57f, 0.0f};
    
    if (tensor_set_data(arm->tensor, arm_positions, sizeof(arm_positions)) < 0) {
        printf("Failed to set arm positions\n");
        return -1;
    }
    
    /* Test 10: Update IMU data */
    float imu_data[9] = {0.0f, 0.0f, 9.81f,  /* accel */
                        0.0f, 0.0f, 0.0f,   /* gyro */
                        0.0f, 1.0f, 0.0f};  /* mag */
    
    if (tensor_set_data(imu->tensor, imu_data, sizeof(imu_data)) < 0) {
        printf("Failed to set IMU data\n");
        return -1;
    }
    
    /* Test 11: Initialize agent cognitive state */
    if (tensor_random(control_agent->cognitive_tensor) < 0) {
        printf("Failed to initialize agent state\n");
        return -1;
    }
    
    /* Test 12: Execute modular processing pipeline */
    workbench_module_t *pipeline[] = {vision_module, control_module, nav_module};
    if (workbench_execute_pipeline(pipeline, 3) < 0) {
        printf("Failed to execute processing pipeline\n");
        return -1;
    }
    
    printf("All test scenarios passed!\n");
    
    /* Print tensor information */
    printf("\nEnhanced Tensor Information:\n");
    printf("Camera tensor (RGB %ux%u @ %.1f fps):\n", 
           camera->tensor->spec.semantics.sensor.width,
           camera->tensor->spec.semantics.sensor.height,
           camera->tensor->spec.semantics.sensor.sampling_rate);
    tensor_print_info(camera->tensor);
    
    printf("\nArm tensor (%u DoF):\n", arm->tensor->spec.semantics.actuator.num_dof);
    tensor_print_info(arm->tensor);
    
    printf("\nIMU tensor (%.1f Hz):\n", imu->tensor->spec.semantics.sensor.sampling_rate);
    tensor_print_info(imu->tensor);
    
    printf("\nAgent cognitive tensor:\n");
    tensor_print_info(control_agent->cognitive_tensor);
    
    printf("\nWorkbench Modules:\n");
    printf("  Vision Module: %s (Type: %d, State: %d)\n", 
           vision_module->name, vision_module->type, vision_module->state);
    printf("  Control Module: %s (Type: %d, State: %d)\n",
           control_module->name, control_module->type, control_module->state);
    printf("  Navigation Module: %s (Type: %d, State: %d)\n",
           nav_module->name, nav_module->type, nav_module->state);
    
    return 0;
}

static int show_status(void) {
    robotics_status_t status;
    
    if (robotics_get_status(&status) < 0) {
        printf("Failed to get system status\n");
        return -1;
    }
    
    printf("Robotics Middleware Status:\n");
    printf("  Initialized: %s\n", status.initialized ? "Yes" : "No");
    printf("  Device count: %u\n", status.device_count);
    printf("  Agent count: %u\n", status.agent_count);
    printf("  Tensor memory usage: %lu bytes\n", status.tensor_memory_usage);
    printf("  Active agents: %u\n", status.active_agents);
    printf("  Uptime: %lu seconds\n", status.uptime_seconds);
    
    return 0;
}

int main(int argc, char *argv[]) {
    int opt;
    const char *config_path = "/etc/robotics/robotics.conf";
    const char *export_file = NULL;
    const char *import_file = NULL;
    bool daemon_mode = false;
    bool show_status_flag = false;
    bool run_tests = false;
    
    static struct option long_options[] = {
        {"config",  required_argument, 0, 'c'},
        {"daemon",  no_argument,       0, 'd'},
        {"export",  required_argument, 0, 'e'},
        {"import",  required_argument, 0, 'i'},
        {"status",  no_argument,       0, 's'},
        {"test",    no_argument,       0, 't'},
        {"help",    no_argument,       0, 'h'},
        {"version", no_argument,       0, 'v'},
        {0, 0, 0, 0}
    };
    
    while ((opt = getopt_long(argc, argv, "c:de:i:sthv", long_options, NULL)) != -1) {
        switch (opt) {
            case 'c':
                config_path = optarg;
                break;
            case 'd':
                daemon_mode = true;
                break;
            case 'e':
                export_file = optarg;
                break;
            case 'i':
                import_file = optarg;
                break;
            case 's':
                show_status_flag = true;
                break;
            case 't':
                run_tests = true;
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            case 'v':
                print_version();
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }
    
    /* Initialize robotics middleware */
    printf("Initializing Robotics Middleware...\n");
    if (robotics_middleware_init(config_path) < 0) {
        fprintf(stderr, "Failed to initialize robotics middleware\n");
        return 1;
    }
    
    /* Handle import if requested */
    if (import_file) {
        printf("Importing system state from: %s\n", import_file);
        if (robotics_import_gguf(import_file) < 0) {
            fprintf(stderr, "Failed to import GGUF file: %s\n", import_file);
            robotics_middleware_cleanup();
            return 1;
        }
    }
    
    /* Run tests if requested */
    if (run_tests) {
        if (run_test_scenarios() < 0) {
            fprintf(stderr, "Test scenarios failed\n");
            robotics_middleware_cleanup();
            return 1;
        }
    }
    
    /* Show status if requested */
    if (show_status_flag) {
        show_status();
    }
    
    /* Handle export if requested */
    if (export_file) {
        printf("Exporting system state to: %s\n", export_file);
        if (robotics_export_gguf(export_file) < 0) {
            fprintf(stderr, "Failed to export GGUF file: %s\n", export_file);
            robotics_middleware_cleanup();
            return 1;
        }
    }
    
    /* If not running as daemon or with one-shot commands, enter interactive mode */
    if (!daemon_mode && !export_file && !import_file && !show_status_flag && !run_tests) {
        printf("Entering interactive mode (Ctrl+C to exit)...\n");
        
        /* Set up signal handlers */
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);
        
        /* Main loop */
        while (g_running) {
            sleep(1);
            /* TODO: Add interactive command processing */
        }
    }
    
    /* Cleanup */
    printf("Shutting down...\n");
    robotics_middleware_cleanup();
    
    return 0;
}