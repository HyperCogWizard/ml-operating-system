#!/usr/bin/env python3
"""
Robotics Engineering Workbench API

This provides a Python API interface for the robotics middleware,
enabling integration with HomeAssistant and other Python-based systems.
"""

import subprocess
import json
import tempfile
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TensorSpec:
    """Enhanced tensor specification for devices and agents"""
    dimensions: int
    shape: List[int]
    dtype: str  # 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32', 'float32', 'float64'
    requires_grad: bool = False
    metadata: str = ""
    
    # Robotics-specific extensions
    modality: str = ""  # 'visual', 'auditory', 'haptic', 'thermal', etc.
    channel_type: str = ""  # 'rgb', 'rgba', 'depth', 'mono', 'stereo', etc.
    num_channels: int = 1
    sampling_rate: float = 0.0
    dof_count: int = 0
    dof_types: List[str] = None
    min_limits: List[float] = None
    max_limits: List[float] = None


@dataclass
class DeviceInfo:
    """Device information"""
    id: int
    name: str
    device_type: str  # 'sensor', 'actuator', 'processor', 'communication', 'power', 'custom'
    tensor_spec: TensorSpec
    active: bool = True


@dataclass
class AgentInfo:
    """Agent information"""
    id: int
    name: str
    cognitive_dimensions: int
    memory_size: int
    attention_heads: int
    embedding_dim: int
    state: str = "idle"
    autonomous: bool = True


@dataclass
class WorkbenchModule:
    """Workbench module information"""
    id: int
    name: str
    module_type: str  # 'sensor', 'actuator', 'processor', 'communication', 'composite'
    description: str
    state: str = "ready"
    capabilities: Dict[str, Any] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = {}
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class RoboticsStatus:
    """System status information"""
    initialized: bool
    device_count: int
    agent_count: int
    tensor_memory_usage: int
    active_agents: int
    uptime_seconds: int
    module_count: int = 0


class RoboticsWorkbench:
    """
    Python API wrapper for the Robotics Engineering Workbench
    
    This class provides high-level Python interfaces for creating devices,
    agents, and managing the robotics middleware system.
    """
    
    def __init__(self, binary_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the robotics workbench
        
        Args:
            binary_path: Path to the robotics-middleware binary
            config_path: Path to configuration file
        """
        if binary_path is None:
            # Try to find the binary in common locations
            candidates = [
                "/usr/bin/robotics-middleware",
                "./robotics-middleware",
                "buildroot-external/package/robotics-middleware/src/robotics-middleware"
            ]
            
            for candidate in candidates:
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    binary_path = candidate
                    break
            
            if binary_path is None:
                raise RuntimeError("Cannot find robotics-middleware binary")
        
        self.binary_path = binary_path
        self.config_path = config_path
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the robotics middleware system
        
        Returns:
            True if initialization successful
        """
        cmd = [self.binary_path, "--status"]
        if self.config_path:
            cmd.extend(["--config", self.config_path])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            self._initialized = (result.returncode == 0)
            return self._initialized
        except subprocess.TimeoutExpired:
            return False
    
    def get_status(self) -> RoboticsStatus:
        """
        Get current system status
        
        Returns:
            RoboticsStatus object with current system state
        """
        cmd = [self.binary_path, "--status"]
        if self.config_path:
            cmd.extend(["--config", self.config_path])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get status: {result.stderr}")
        
        # Parse the status output (simplified parsing)
        status_data = {}
        for line in result.stdout.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                
                # Convert values to appropriate types
                if key == 'initialized':
                    status_data[key] = value == 'Yes'
                elif 'count' in key or 'usage' in key or 'uptime' in key:
                    # Extract numbers from strings like "5 bytes" or "10 seconds"
                    numbers = ''.join(filter(str.isdigit, value))
                    status_data[key] = int(numbers) if numbers else 0
                else:
                    status_data[key] = value
        
        return RoboticsStatus(
            initialized=status_data.get('initialized', False),
            device_count=status_data.get('device_count', 0),
            agent_count=status_data.get('agent_count', 0),
            tensor_memory_usage=status_data.get('tensor_memory_usage', 0),
            active_agents=status_data.get('active_agents', 0),
            uptime_seconds=status_data.get('uptime_seconds', 0),
            module_count=status_data.get('module_count', 0)
        )
    
    def run_test_scenarios(self) -> bool:
        """
        Run built-in test scenarios
        
        Returns:
            True if tests passed
        """
        cmd = [self.binary_path, "--test"]
        if self.config_path:
            cmd.extend(["--config", self.config_path])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and "All test scenarios passed!" in result.stdout
    
    def export_gguf(self, filepath: str) -> bool:
        """
        Export current system state to GGUF file
        
        Args:
            filepath: Path where to save the GGUF file
            
        Returns:
            True if export successful
        """
        cmd = [self.binary_path, "--export", filepath]
        if self.config_path:
            cmd.extend(["--config", self.config_path])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0 and os.path.exists(filepath)
    
    def import_gguf(self, filepath: str) -> bool:
        """
        Import system state from GGUF file
        
        Args:
            filepath: Path to the GGUF file to import
            
        Returns:
            True if import successful
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"GGUF file not found: {filepath}")
        
        cmd = [self.binary_path, "--import", filepath]
        if self.config_path:
            cmd.extend(["--config", self.config_path])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    
    def create_camera_device(self, name: str, width: int = 640, height: int = 480,
                           channel_type: str = "rgb", fps: float = 30.0) -> DeviceInfo:
        """
        Create a camera sensor device with enhanced semantics
        
        Args:
            name: Device name
            width: Image width
            height: Image height
            channel_type: Channel type ('rgb', 'rgba', 'depth', 'mono')
            fps: Frames per second
            
        Returns:
            DeviceInfo for the created device
        """
        channels = {'rgb': 3, 'rgba': 4, 'depth': 1, 'mono': 1}.get(channel_type, 3)
        
        tensor_spec = TensorSpec(
            dimensions=3,
            shape=[height, width, channels],
            dtype='uint8',
            metadata=f"camera_{width}x{height}_{channels}ch_{fps}fps",
            modality='visual',
            channel_type=channel_type,
            num_channels=channels,
            sampling_rate=fps
        )
        
        return DeviceInfo(
            id=0,  # Would be assigned by the system
            name=name,
            device_type='sensor',
            tensor_spec=tensor_spec
        )
    
    def create_robot_arm_device(self, name: str, dof: int = 6, 
                              dof_types: List[str] = None,
                              min_limits: List[float] = None,
                              max_limits: List[float] = None) -> DeviceInfo:
        """
        Create a robot arm actuator device with enhanced DoF semantics
        
        Args:
            name: Device name
            dof: Degrees of freedom (number of joints)
            dof_types: Types for each DoF ('rotational', 'linear', etc.)
            min_limits: Minimum limits for each DoF
            max_limits: Maximum limits for each DoF
            
        Returns:
            DeviceInfo for the created device
        """
        if dof_types is None:
            dof_types = ['rotational'] * dof
        if min_limits is None:
            min_limits = [-3.14159] * dof
        if max_limits is None:
            max_limits = [3.14159] * dof
            
        tensor_spec = TensorSpec(
            dimensions=1,
            shape=[dof],
            dtype='float32',
            metadata=f"robot_arm_{dof}dof",
            dof_count=dof,
            dof_types=dof_types,
            min_limits=min_limits,
            max_limits=max_limits
        )
        
        return DeviceInfo(
            id=1,  # Would be assigned by the system
            name=name,
            device_type='actuator',
            tensor_spec=tensor_spec
        )
    
    def create_control_agent(self, name: str, memory_size: int = 1024, 
                           attention_heads: int = 8, embedding_dim: int = 512) -> AgentInfo:
        """
        Create a control agent
        
        Args:
            name: Agent name
            memory_size: Memory size for the agent
            attention_heads: Number of attention heads
            embedding_dim: Embedding dimension
            
        Returns:
            AgentInfo for the created agent
        """
        return AgentInfo(
            id=0,  # Would be assigned by the system
            name=name,
            cognitive_dimensions=3,
            memory_size=memory_size,
            attention_heads=attention_heads,
            embedding_dim=embedding_dim
        )
    
    def create_workbench_module(self, name: str, module_type: str, description: str = "") -> WorkbenchModule:
        """
        Create a workbench module
        
        Args:
            name: Module name
            module_type: Module type ('sensor', 'actuator', 'processor', 'communication', 'composite')
            description: Module description
            
        Returns:
            WorkbenchModule for the created module
        """
        capabilities = {}
        if module_type == 'sensor':
            capabilities = {'supports_streaming': True, 'max_output_tensors': 4}
        elif module_type == 'actuator':
            capabilities = {'supports_configuration': True, 'max_input_tensors': 4}
        elif module_type == 'processor':
            capabilities = {'supports_prediction': True, 'supports_learning': True, 
                          'max_input_tensors': 8, 'max_output_tensors': 8}
        
        return WorkbenchModule(
            id=0,  # Would be assigned by the system
            name=name,
            module_type=module_type,
            description=description,
            capabilities=capabilities
        )
    
    def create_vision_module(self, name: str) -> WorkbenchModule:
        """Create a vision processing module"""
        return self.create_workbench_module(name, 'processor', 'Computer vision processing module')
    
    def create_control_module(self, name: str) -> WorkbenchModule:
        """Create a control processing module"""
        return self.create_workbench_module(name, 'processor', 'Robot control processing module')
    
    def create_navigation_module(self, name: str) -> WorkbenchModule:
        """Create a navigation processing module"""
        return self.create_workbench_module(name, 'processor', 'Robot navigation processing module')
    
    def create_manipulation_module(self, name: str) -> WorkbenchModule:
        """Create a manipulation processing module"""
        return self.create_workbench_module(name, 'processor', 'Robot manipulation processing module')
    
    def create_imu_device(self, name: str, sampling_rate: float = 100.0) -> DeviceInfo:
        """
        Create an IMU sensor device
        
        Args:
            name: Device name
            sampling_rate: Sampling rate in Hz
            
        Returns:
            DeviceInfo for the created device
        """
        tensor_spec = TensorSpec(
            dimensions=1,
            shape=[9],  # 3 accel + 3 gyro + 3 mag
            dtype='float32',
            metadata=f"imu_9axis_{sampling_rate}Hz",
            modality='inertial',
            channel_type='multichannel',
            num_channels=9,
            sampling_rate=sampling_rate
        )
        
        return DeviceInfo(
            id=2,  # Would be assigned by the system
            name=name,
            device_type='sensor',
            tensor_spec=tensor_spec
        )


class HomeAssistantIntegration:
    """
    Integration layer for HomeAssistant
    
    This class provides methods to transform HomeAssistant entities
    into robotics middleware components and vice versa.
    """
    
    def __init__(self, workbench: RoboticsWorkbench):
        """
        Initialize HomeAssistant integration
        
        Args:
            workbench: RoboticsWorkbench instance
        """
        self.workbench = workbench
        self.entity_mappings = {}
    
    def map_sensor_entity(self, entity_id: str, sensor_type: str = "generic") -> DeviceInfo:
        """
        Map a HomeAssistant sensor entity to a robotics device with enhanced semantics
        
        Args:
            entity_id: HomeAssistant entity ID (e.g., "sensor.temperature")
            sensor_type: Type of sensor
            
        Returns:
            DeviceInfo for the mapped device
        """
        if sensor_type == "camera":
            device = self.workbench.create_camera_device(f"ha_{entity_id}", 640, 480, "rgb", 30.0)
        elif sensor_type == "temperature":
            # Temperature sensor with enhanced semantics
            tensor_spec = TensorSpec(
                dimensions=1,
                shape=[1],
                dtype='float32',
                metadata=f"homeassistant_{entity_id}_temperature",
                modality='thermal',
                channel_type='mono',
                num_channels=1,
                sampling_rate=1.0  # Assume 1 Hz sampling
            )
            device = DeviceInfo(
                id=len(self.entity_mappings),
                name=f"ha_{entity_id}",
                device_type='sensor',
                tensor_spec=tensor_spec
            )
        else:
            # Generic sensor with single value
            tensor_spec = TensorSpec(
                dimensions=1,
                shape=[1],
                dtype='float32',
                metadata=f"homeassistant_{entity_id}_{sensor_type}",
                modality='custom',
                channel_type='mono',
                num_channels=1
            )
            device = DeviceInfo(
                id=len(self.entity_mappings),
                name=f"ha_{entity_id}",
                device_type='sensor',
                tensor_spec=tensor_spec
            )
        
        self.entity_mappings[entity_id] = device
        return device
    
    def map_switch_entity(self, entity_id: str) -> DeviceInfo:
        """
        Map a HomeAssistant switch entity to a robotics actuator
        
        Args:
            entity_id: HomeAssistant entity ID (e.g., "switch.light")
            
        Returns:
            DeviceInfo for the mapped device
        """
        tensor_spec = TensorSpec(
            dimensions=1,
            shape=[1],
            dtype='uint8',  # 0 = off, 1 = on
            metadata=f"homeassistant_{entity_id}_switch"
        )
        
        device = DeviceInfo(
            id=len(self.entity_mappings),
            name=f"ha_{entity_id}",
            device_type='actuator',
            tensor_spec=tensor_spec
        )
        
        self.entity_mappings[entity_id] = device
        return device
    
    def create_automation_agent(self, automation_name: str, entities: List[str]) -> AgentInfo:
        """
        Create an agent to replace a HomeAssistant automation
        
        Args:
            automation_name: Name of the automation
            entities: List of entity IDs involved in the automation
            
        Returns:
            AgentInfo for the automation agent
        """
        agent = self.workbench.create_control_agent(
            f"automation_{automation_name}",
            memory_size=len(entities) * 64,  # Scale memory with entity count
            attention_heads=min(8, len(entities)),  # Scale attention heads
            embedding_dim=256
        )
        
        return agent
    
    def get_entity_mappings(self) -> Dict[str, DeviceInfo]:
        """
        Get all current entity mappings
        
        Returns:
            Dictionary mapping entity IDs to DeviceInfo objects
        """
        return self.entity_mappings.copy()


def main():
    """
    Example usage of the Robotics Engineering Workbench API
    """
    print("Robotics Engineering Workbench API Demo")
    print("=" * 50)
    
    # Initialize workbench
    try:
        workbench = RoboticsWorkbench()
        print(f"Using binary: {workbench.binary_path}")
        
        if not workbench.initialize():
            print("Failed to initialize workbench")
            return
        
        print("✓ Workbench initialized successfully")
        
        # Get status
        status = workbench.get_status()
        print(f"✓ Status: {status.device_count} devices, {status.agent_count} agents")
        
        # Run test scenarios
        if workbench.run_test_scenarios():
            print("✓ Test scenarios completed successfully")
        else:
            print("✗ Test scenarios failed")
        
        # Test GGUF export/import
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            temp_file = f.name
        
        try:
            if workbench.export_gguf(temp_file):
                print(f"✓ Exported system state to: {temp_file}")
                file_size = os.path.getsize(temp_file) / (1024 * 1024)
                print(f"  File size: {file_size:.2f} MB")
                
                if workbench.import_gguf(temp_file):
                    print("✓ Successfully imported system state")
                else:
                    print("✗ Failed to import system state")
            else:
                print("✗ Failed to export system state")
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Demo HomeAssistant integration
        print("\nHomeAssistant Integration Demo:")
        print("-" * 30)
        
        ha_integration = HomeAssistantIntegration(workbench)
        
        # Map some HomeAssistant entities with enhanced semantics
        temp_sensor = ha_integration.map_sensor_entity("sensor.living_room_temperature", "temperature")
        camera_device = ha_integration.map_sensor_entity("camera.front_door", "camera")
        light_switch = ha_integration.map_switch_entity("switch.living_room_light")
        
        print(f"✓ Mapped temperature sensor: {temp_sensor.name}")
        print(f"✓ Mapped camera device: {camera_device.name} ({camera_device.tensor_spec.modality})")
        print(f"✓ Mapped light switch: {light_switch.name}")
        
        # Demonstrate enhanced device creation
        print("\nEnhanced Device Creation:")
        print("-" * 30)
        
        # Create devices with rich semantic specifications
        hd_camera = workbench.create_camera_device("hd_camera", 1920, 1080, "rgb", 60.0)
        robot_arm = workbench.create_robot_arm_device("manipulator", 7, 
                                                     ['rotational'] * 6 + ['linear'],
                                                     [-3.14] * 6 + [0.0],
                                                     [3.14] * 6 + [0.1])
        imu_sensor = workbench.create_imu_device("main_imu", 200.0)
        
        print(f"✓ Created HD camera: {hd_camera.tensor_spec.shape} @ {hd_camera.tensor_spec.sampling_rate} fps")
        print(f"✓ Created robot arm: {robot_arm.tensor_spec.dof_count} DoF, types: {robot_arm.tensor_spec.dof_types}")
        print(f"✓ Created IMU: {imu_sensor.tensor_spec.num_channels} channels @ {imu_sensor.tensor_spec.sampling_rate} Hz")
        
        # Demonstrate workbench modules
        print("\nWorkbench Modules:")
        print("-" * 30)
        
        vision_mod = workbench.create_vision_module("advanced_vision")
        control_mod = workbench.create_control_module("arm_controller")
        nav_mod = workbench.create_navigation_module("smart_nav")
        manip_mod = workbench.create_manipulation_module("gripper_control")
        
        print(f"✓ Vision module: {vision_mod.name} ({vision_mod.module_type})")
        print(f"  Capabilities: {vision_mod.capabilities}")
        print(f"✓ Control module: {control_mod.name} ({control_mod.module_type})")
        print(f"✓ Navigation module: {nav_mod.name} ({nav_mod.module_type})")
        print(f"✓ Manipulation module: {manip_mod.name} ({manip_mod.module_type})")
        
        # Create automation agent
        automation_agent = ha_integration.create_automation_agent(
            "motion_light_control",
            ["sensor.motion_detector", "switch.living_room_light"]
        )
        print(f"✓ Created automation agent: {automation_agent.name}")
        
        print(f"\nTotal mapped entities: {len(ha_integration.get_entity_mappings())}")
        print(f"Total created modules: 4")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())