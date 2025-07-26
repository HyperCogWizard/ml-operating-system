# Robotics Engineering Workbench

**A Practical Robotics Engineering Workbench Using GGUF for Marduk's Robotics Lab**

This package extends the Home Assistant Operating System with a comprehensive robotics middleware abstraction, providing tensor-based device management, neural-symbolic agent architectures, and GGUF-based state persistence.

## Overview

The Robotics Engineering Workbench transforms traditional static automation into a dynamic, agentic system where:

- **Devices** are abstracted as multi-dimensional tensors
- **Agents** possess cognitive architectures with memory, attention, and embedding layers
- **System state** is persistable via GGUF format for distributed cognition
- **HomeAssistant entities** are kernelized into tensor nodes
- **Scheme functions** provide neural-symbolic middleware capabilities

## Architecture

```
[Robotics Middleware Abstraction]
     |
     v
[Engineering Workbench]
     |  (Integrate simulation, device I/O, agent configuration)
     v
[GGUF Integration Layer]
     |  (Tensor serialization, kernel embedding, agent state export)
     v
[HomeAssistant Transformation]
     |  (Automate device orchestration, sensor fusion, actuator control)
     v
[Marduk's Robotics Lab]
     (Distributed agentic cognition, neural-symbolic middleware)
```

## Core Components

### 1. Tensor Abstraction Layer
- **Multi-dimensional tensors** for device I/O representation
- **Type safety** with uint8, int8, uint16, int16, uint32, int32, float32, float64
- **Memory management** with reference counting and thread safety
- **Operations** including addition, zero initialization, random initialization

### 2. Hypergraph Structure
- **Device nodes** representing sensors, actuators, processors, communication, power
- **Agent nodes** with cognitive architectures
- **Dynamic connections** between devices and agents
- **Thread-safe operations** with read-write locks

### 3. GGUF Integration
- **Complete state serialization** of all tensors and metadata
- **Version 3 GGUF format** with 32-byte alignment
- **Round-trip fidelity** for system state persistence
- **Distributed cognition** support via state sharing

### 4. Agent Architecture
- **3D Cognitive tensors** (memory × attention heads × embedding dimension)
- **State management** (idle, active, learning, planning, executing, error)
- **Autonomous operation** with self-modification capabilities
- **Scheme integration** for neural-symbolic processing

## Device Types and Tensor Specifications

### Camera Sensor
```c
tensor_spec_t camera_spec = {
    .dimensions = 2,
    .shape = {640, 480},        // Width × Height
    .dtype = TENSOR_UINT8,      // RGB pixel values
    .metadata = "camera_rgb"
};
```

### Robot Arm Actuator
```c
tensor_spec_t arm_spec = {
    .dimensions = 1,
    .shape = {6},               // 6 degrees of freedom
    .dtype = TENSOR_FLOAT32,    // Joint positions in radians
    .metadata = "robot_arm_joints"
};
```

### Agent Cognitive State
```c
tensor_spec_t cognitive_spec = {
    .dimensions = 3,
    .shape = {1024, 8, 512},    // Memory × Attention × Embedding
    .dtype = TENSOR_FLOAT32,    // Neural activations
    .metadata = "cognitive_state"
};
```

## API Usage

### C Library
```c
#include "robotics_middleware.h"

// Initialize system
robotics_middleware_init("/etc/robotics/robotics.conf");

// Create camera device
tensor_spec_t camera_spec = {2, {640, 480}, TENSOR_UINT8, false, "camera"};
device_node_t *camera = robotics_create_device("main_camera", DEVICE_TYPE_SENSOR, &camera_spec);

// Create control agent
agent_config_t agent_config = {3, 1024, 8, 512, "(define control-loop ...)", "transformer"};
agent_t *agent = robotics_create_agent("controller", &agent_config);

// Export system state
robotics_export_gguf("/var/lib/robotics/state.gguf");
```

### Python Workbench API
```python
from workbench_api import RoboticsWorkbench, HomeAssistantIntegration

# Initialize workbench
workbench = RoboticsWorkbench()
workbench.initialize()

# Create devices
camera = workbench.create_camera_device("front_door", 1920, 1080)
arm = workbench.create_robot_arm_device("manipulator", dof=7)

# HomeAssistant integration
ha = HomeAssistantIntegration(workbench)
temp_sensor = ha.map_sensor_entity("sensor.living_room_temperature")
light_agent = ha.create_automation_agent("motion_light", ["sensor.motion", "light.living_room"])

# Export/import state
workbench.export_gguf("/tmp/lab_state.gguf")
workbench.import_gguf("/tmp/lab_state.gguf")
```

### Scheme Neural-Symbolic Layer
```scheme
(define (cognitive-control-loop agent system-state)
  "Main cognitive control loop for autonomous agents"
  (let ((sensor-inputs (get-sensor-inputs system-state agent)))
    (cond
      ((eq? (agent-state agent) 'idle)
       (if (any-sensor-active? sensor-inputs)
           (transition-to-state agent 'active)))
      ((eq? (agent-state agent) 'active)
       (let ((processed (sensory-processing sensor-inputs agent)))
         (execute-motor-commands (motor-control processed agent)))))))

;; HomeAssistant automation transformation
(define automation-agent 
  (automation->agent "motion_light"
                     '("sensor.motion" "light.living_room")
                     '((motion-detected . #t))
                     '((turn-on-light . "light.living_room"))))
```

## Configuration

### System Configuration (`/etc/robotics/robotics.conf`)
```ini
# Memory settings
tensor_memory_limit = 1073741824  # 1GB

# Device and agent limits  
max_devices = 1024
max_agents = 256

# GGUF integration
gguf_version = 3.0
auto_export_interval = 300  # 5 minutes

# Neural-symbolic middleware
scheme_interpreter = guile
scheme_modules_path = /usr/lib/robotics/scheme

# HomeAssistant integration
homeassistant_api_url = http://supervisor/core/api
homeassistant_token_file = /etc/robotics/ha_token

# Marduk's Lab settings
lab_name = "Marduk's Robotics Lab"
distributed_cognition = true
p_system_support = true
```

### SystemD Service
```ini
[Unit]
Description=Robotics Middleware for Engineering Workbench
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/robotics-middleware --daemon --config /etc/robotics/robotics.conf
Restart=always
User=root

[Install]
WantedBy=multi-user.target
```

## HomeAssistant Integration

### Entity Mapping
The system automatically converts HomeAssistant entities to tensor-based devices:

- **Sensors** → 1D float32 tensors (scalar values)
- **Cameras** → 2D uint8 tensors (image data)
- **Switches** → 1D uint8 tensors (binary states)
- **Lights** → 1D/3D uint8 tensors (brightness/RGB)

### Automation Kernelization
Traditional HomeAssistant automations are transformed into autonomous agents:

```yaml
# Traditional automation
automation:
  - alias: "Motion Light"
    trigger:
      platform: state
      entity_id: binary_sensor.motion
      to: "on"
    action:
      service: light.turn_on
      entity_id: light.living_room
```

Becomes an agent with cognitive architecture:

```c
agent_config_t motion_agent = {
    .cognitive_dimensions = 3,
    .memory_size = 128,           // Scaled to automation complexity
    .attention_heads = 4,
    .embedding_dim = 256,
    .scheme_functions = "(define motion-light-control ...)"
};
```

## GGUF State Persistence

### Export Format
```
GGUF Header:
  magic: "GGUF" (0x46554747)
  version: 3
  tensor_count: N
  metadata_kv_count: M

Tensor Info Array:
  - name: "device_0_main_camera"
    dimensions: 2
    shape: [640, 480]
    type: UINT8
    offset: data_offset
  
  - name: "agent_0_controller"  
    dimensions: 3
    shape: [1024, 8, 512]
    type: FLOAT32
    offset: data_offset + previous_size

Data Section:
  [aligned tensor data]
```

### Distributed Cognition
GGUF files enable distributed cognition across multiple systems:

1. **Export** system state on Device A
2. **Transfer** GGUF file to Device B
3. **Import** and continue processing on Device B
4. **Sync** cognitive states across the network

## Testing

### Build and Test
```bash
cd buildroot-external/package/robotics-middleware/src
make clean && make
./robotics-middleware --test
./robotics-middleware --test --export /tmp/test.gguf
./robotics-middleware --import /tmp/test.gguf --status
```

### Python API Tests
```bash
cd tests/robotics_middleware
python3 test_robotics_middleware.py
python3 -c "from workbench_api import *; print('API imported successfully')"
```

### Integration with BuildRoot
```bash
make generic_x86_64-config
make menuconfig  # Enable BR2_PACKAGE_ROBOTICS_MIDDLEWARE
make
```

## Marduk's Robotics Lab Features

### P-System Membrane Computing
- **Membrane structures** for agent isolation and communication
- **Object evolution** within membranes
- **Rule-based processing** for membrane interactions

### Recursive Agent Modification
- **Self-modifying architectures** that can evolve their own cognitive structures
- **Dynamic Scheme function generation** for new behaviors
- **Tensor architecture updates** based on environmental needs

### Distributed Agentic Cognition
- **Multi-device coordination** via GGUF state sharing
- **Shared memory spaces** across physical systems
- **Cognitive load balancing** for complex tasks

## Performance Characteristics

- **Memory usage**: ~17MB for basic test scenario (camera + arm + agent)
- **Tensor operations**: Thread-safe with mutex protection
- **GGUF I/O**: ~50MB/s on typical storage
- **Agent cognitive cycles**: Configurable, typically 10-100Hz
- **Device polling**: 1ms default interval for real-time response

## Security Considerations

- **SystemD hardening**: NoNewPrivileges, ProtectSystem=strict
- **Memory isolation**: Per-tensor reference counting
- **Configuration validation**: Input sanitization for all parameters
- **GGUF verification**: Magic number and version checks

## Future Extensions

1. **GPU acceleration** for tensor operations
2. **Real-time device drivers** for industrial robotics
3. **WebAssembly agents** for sandboxed execution
4. **ROS2 bridge** for compatibility with existing robotics stacks
5. **Neural network compilation** to GGUF format
6. **Distributed training** across multiple HomeAssistant instances

## Troubleshooting

### Common Issues

**Q: Binary crashes with segmentation fault**
A: Check memory limits in config, ensure proper tensor cleanup

**Q: GGUF export fails**
A: Verify write permissions to export directory

**Q: HomeAssistant entities not mapping**
A: Check entity IDs and API token configuration

**Q: Agent cognitive loops not responding** 
A: Increase memory_size or reduce attention_heads in agent config

### Debug Mode
```bash
robotics-middleware --config /etc/robotics/robotics.conf --status
export ROBOTICS_DEBUG=1
robotics-middleware --test
```

### Log Analysis
```bash
journalctl -u robotics-middleware.service -f
tail -f /var/log/robotics-middleware.log
```

---

**License**: Apache 2.0  
**Maintainer**: Marduk's Robotics Lab  
**Version**: 1.0.0