#!/usr/bin/env python3
"""
Tests for Robotics Middleware Engineering Workbench

This test suite validates the core functionality of the robotics middleware
including tensor operations, GGUF integration, device management, and
agent orchestration.
"""

import pytest
import subprocess
import tempfile
import os
import json
import time
from pathlib import Path


class TestRoboticsMiddleware:
    """Test class for robotics middleware functionality"""
    
    @pytest.fixture
    def middleware_binary(self):
        """Path to the robotics middleware binary"""
        return Path(__file__).parent.parent.parent / "buildroot-external" / "package" / "robotics-middleware" / "src" / "robotics-middleware"
    
    @pytest.fixture
    def temp_gguf_file(self):
        """Temporary GGUF file for testing"""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            yield f.name
        os.unlink(f.name)
    
    def test_version_command(self, middleware_binary):
        """Test that version command works"""
        result = subprocess.run([str(middleware_binary), "--version"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert "Robotics Middleware v1.0.0" in result.stdout
        assert "GGUF Integration for Marduk's Robotics Lab" in result.stdout
    
    def test_help_command(self, middleware_binary):
        """Test that help command works"""
        result = subprocess.run([str(middleware_binary), "--help"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert "Robotics Engineering Workbench" in result.stdout
        assert "--config" in result.stdout
        assert "--export" in result.stdout
        assert "--import" in result.stdout
    
    def test_status_command(self, middleware_binary):
        """Test status command functionality"""
        result = subprocess.run([str(middleware_binary), "--status"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert "Robotics Middleware Status:" in result.stdout
        assert "Initialized: Yes" in result.stdout
        assert "Device count:" in result.stdout
        assert "Agent count:" in result.stdout
    
    def test_test_scenarios(self, middleware_binary):
        """Test the built-in test scenarios"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert "Running test scenarios..." in result.stdout
        assert "All test scenarios passed!" in result.stdout
        assert "Created device: main_camera" in result.stdout
        assert "Created device: robot_arm" in result.stdout
        assert "Created agent: main_controller" in result.stdout
    
    def test_gguf_export(self, middleware_binary, temp_gguf_file):
        """Test GGUF export functionality"""
        result = subprocess.run([str(middleware_binary), "--test", "--export", temp_gguf_file], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        assert f"Exported system state to GGUF: {temp_gguf_file}" in result.stdout
        assert os.path.exists(temp_gguf_file)
        
        # Check file size is reasonable (should contain tensor data)
        file_size = os.path.getsize(temp_gguf_file)
        assert file_size > 1000000  # Should be > 1MB due to tensor data
    
    def test_gguf_import(self, middleware_binary, temp_gguf_file):
        """Test GGUF import functionality"""
        # First export some data
        export_result = subprocess.run([str(middleware_binary), "--test", "--export", temp_gguf_file], 
                                     capture_output=True, text=True)
        assert export_result.returncode == 0
        
        # Then import it
        import_result = subprocess.run([str(middleware_binary), "--import", temp_gguf_file], 
                                     capture_output=True, text=True)
        assert import_result.returncode == 0
        assert f"Imported system state from GGUF: {temp_gguf_file}" in import_result.stdout
        assert "Successfully read GGUF file" in import_result.stdout
        assert "3 tensors" in import_result.stdout
    
    def test_enhanced_tensor_semantics(self, middleware_binary):
        """Test enhanced tensor semantics with DoF, channels, and modalities"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Camera should have visual modality with RGB channels
        camera_output = result.stdout
        assert "Camera tensor (RGB 640x480 @ 30.0 fps):" in camera_output
        
        # Robot arm should have DoF information  
        assert "Arm tensor (6 DoF):" in camera_output
        
        # IMU should have sampling rate information
        assert "IMU tensor (100.0 Hz):" in camera_output
        
        # Check that devices are created with correct types
        assert "Created device: main_camera (ID: 0, Type: 0)" in result.stdout  # SENSOR
        assert "Created device: robot_arm (ID: 1, Type: 1)" in result.stdout     # ACTUATOR  
        assert "Created device: main_imu (ID: 2, Type: 0)" in result.stdout      # SENSOR
    
    def test_hypergraph_functionality(self, middleware_binary):
        """Test enhanced hypergraph device, agent, and module management"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check hypergraph operations for devices
        assert "Added device to hypergraph: main_camera (total: 1)" in result.stdout
        assert "Added device to hypergraph: robot_arm (total: 2)" in result.stdout
        assert "Added device to hypergraph: main_imu (total: 3)" in result.stdout
        
        # Check hypergraph operations for agents
        assert "Added agent to hypergraph: main_controller (total: 1)" in result.stdout
        
        # Check hypergraph cleanup
        assert "Hypergraph cleaned up" in result.stdout
    
    def test_enhanced_device_creation(self, middleware_binary):
        """Test enhanced device creation with semantic specifications"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check enhanced camera device with RGB channels and FPS
        assert "Camera tensor (RGB 640x480 @ 30.0 fps):" in result.stdout
        assert "Dimensions: 3" in result.stdout
        assert "Shape: [480, 640, 3]" in result.stdout  # Height x Width x Channels
        assert "Size: 921600 bytes" in result.stdout  # 480 * 640 * 3
        assert "Metadata: camera_640x480_3ch_30.0fps" in result.stdout
        
        # Check enhanced robot arm with DoF specifications
        assert "Arm tensor (6 DoF):" in result.stdout
        assert "Shape: [6]" in result.stdout
        assert "Metadata: robot_arm_6dof" in result.stdout
        
        # Check IMU sensor with sampling rate
        assert "IMU tensor (100.0 Hz):" in result.stdout
        assert "Shape: [9]" in result.stdout  # 3 accel + 3 gyro + 3 mag
        assert "Metadata: imu_9axis_100.0Hz" in result.stdout
    
    def test_workbench_modules(self, middleware_binary):
        """Test workbench module creation and management"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check workbench modules are created
        assert "Created workbench module: vision_processor (ID: 0, Type: 2)" in result.stdout
        assert "Created workbench module: arm_controller (ID: 1, Type: 2)" in result.stdout
        assert "Created workbench module: navigator (ID: 2, Type: 2)" in result.stdout
        
        # Check modules are registered
        assert "Added module to registry: vision_processor (total: 1)" in result.stdout
        assert "Added module to registry: arm_controller (total: 2)" in result.stdout
        assert "Added module to registry: navigator (total: 3)" in result.stdout
        
        # Check module connections
        assert "Connected modules: vision_processor -> arm_controller" in result.stdout
        assert "Connected modules: vision_processor -> navigator" in result.stdout
        
        # Check pipeline execution
        assert "Executed pipeline with 3 modules" in result.stdout
        
        # Check module information in output
        assert "Vision Module: vision_processor (Type: 2, State: 2)" in result.stdout
        assert "Control Module: arm_controller (Type: 2, State: 2)" in result.stdout
        assert "Navigation Module: navigator (Type: 2, State: 2)" in result.stdout
        
    def test_memory_management(self, middleware_binary):
        """Test memory management by running multiple operations"""
        # Run test multiple times to check for memory leaks
        for _ in range(3):
            result = subprocess.run([str(middleware_binary), "--test"], 
                                  capture_output=True, text=True)
            assert result.returncode == 0
            assert "Robotics Middleware cleaned up" in result.stdout


class TestGGUFIntegration:
    """Test GGUF-specific functionality"""
    
    @pytest.fixture
    def middleware_binary(self):
        """Path to the robotics middleware binary"""
        return Path(__file__).parent.parent.parent / "buildroot-external" / "package" / "robotics-middleware" / "src" / "robotics-middleware"
    
    def test_gguf_round_trip(self, middleware_binary):
        """Test complete GGUF export/import round trip"""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            temp_file = f.name
        
        try:
            # Export data
            export_result = subprocess.run([str(middleware_binary), "--test", "--export", temp_file], 
                                         capture_output=True, text=True)
            assert export_result.returncode == 0
            
            # Verify file exists and has reasonable size
            assert os.path.exists(temp_file)
            file_size = os.path.getsize(temp_file)
            assert file_size > 1000000  # > 1MB
            
            # Import data
            import_result = subprocess.run([str(middleware_binary), "--import", temp_file, "--status"], 
                                         capture_output=True, text=True)
            assert import_result.returncode == 0
            assert "Successfully read GGUF file" in import_result.stdout
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_gguf_tensor_metadata(self, middleware_binary):
        """Test that tensor metadata is preserved in GGUF"""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            temp_file = f.name
        
        try:
            # Export and import
            subprocess.run([str(middleware_binary), "--test", "--export", temp_file], 
                          capture_output=True, text=True)
            
            result = subprocess.run([str(middleware_binary), "--import", temp_file], 
                                  capture_output=True, text=True)
            assert result.returncode == 0
            
            # Check tensor names are preserved
            assert "device_0_main_camera" in result.stdout
            assert "device_1_robot_arm" in result.stdout
            assert "agent_0_main_controller" in result.stdout
            
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestRoboticsWorkbench:
    """Test robotics engineering workbench functionality"""
    
    @pytest.fixture
    def middleware_binary(self):
        """Path to the robotics middleware binary"""
        return Path(__file__).parent.parent.parent / "buildroot-external" / "package" / "robotics-middleware" / "src" / "robotics-middleware"
    
    def test_sensor_device_abstraction(self, middleware_binary):
        """Test sensor device tensor abstraction with enhanced semantics"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Camera sensor should have 3D tensor for RGB image data
        assert "Camera tensor (RGB 640x480 @ 30.0 fps):" in result.stdout
        assert "Dimensions: 3" in result.stdout
        assert "Shape: [480, 640, 3]" in result.stdout  # Height x Width x Channels
        assert "Data type: 0" in result.stdout  # UINT8 for image
        assert "Size: 921600 bytes" in result.stdout  # 480 * 640 * 3
        
        # IMU sensor should have 1D tensor for 9-axis data
        assert "IMU tensor (100.0 Hz):" in result.stdout
        assert "Dimensions: 1" in result.stdout
        assert "Shape: [9]" in result.stdout  # 3 accel + 3 gyro + 3 mag
        assert "Data type: 6" in result.stdout  # FLOAT32 for sensor data
    
    def test_actuator_device_abstraction(self, middleware_binary):
        """Test actuator device tensor abstraction with DoF semantics"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Robot arm should have 1D tensor for joint positions with DoF info
        assert "Arm tensor (6 DoF):" in result.stdout
        assert "Dimensions: 1" in result.stdout
        assert "Shape: [6]" in result.stdout  # 6-DOF arm
        assert "Data type: 6" in result.stdout  # FLOAT32 for positions
        assert "Metadata: robot_arm_6dof" in result.stdout
    
    def test_agent_cognitive_architecture(self, middleware_binary):
        """Test agent cognitive tensor architecture"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Agent should have 3D cognitive tensor
        assert "Agent cognitive tensor:" in result.stdout
        assert "Dimensions: 3" in result.stdout
        assert "Shape: [1024, 8, 512]" in result.stdout  # memory x attention x embedding
        assert "Data type: 6" in result.stdout  # FLOAT32 for neural data
    
    def test_modular_workbench_composition(self, middleware_binary):
        """Test modular workbench component composition"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check that modules can be composed into pipelines
        assert "Connected modules: vision_processor -> arm_controller" in result.stdout
        assert "Connected modules: vision_processor -> navigator" in result.stdout
        assert "Executed pipeline with 3 modules" in result.stdout
        
        # Check module states after execution
        assert "Vision Module: vision_processor (Type: 2, State: 2)" in result.stdout
        assert "Control Module: arm_controller (Type: 2, State: 2)" in result.stdout
        assert "Navigation Module: navigator (Type: 2, State: 2)" in result.stdout


class TestWorkbenchModules:
    """Test workbench module system specifically"""
    
    @pytest.fixture
    def middleware_binary(self):
        """Path to the robotics middleware binary"""
        return Path(__file__).parent.parent.parent / "buildroot-external" / "package" / "robotics-middleware" / "src" / "robotics-middleware"
    
    def test_module_creation_and_registry(self, middleware_binary):
        """Test module creation and registry management"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check modules are created with correct IDs and types
        assert "Created workbench module: vision_processor (ID: 0, Type: 2)" in result.stdout
        assert "Created workbench module: arm_controller (ID: 1, Type: 2)" in result.stdout
        assert "Created workbench module: navigator (ID: 2, Type: 2)" in result.stdout
        
        # Check modules are added to registry
        assert "Added module to registry: vision_processor (total: 1)" in result.stdout
        assert "Added module to registry: arm_controller (total: 2)" in result.stdout
        assert "Added module to registry: navigator (total: 3)" in result.stdout
    
    def test_module_connection_graph(self, middleware_binary):
        """Test module connection and hypergraph composition"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check module connections are established
        assert "Connected modules: vision_processor -> arm_controller" in result.stdout
        assert "Connected modules: vision_processor -> navigator" in result.stdout
        
        # This demonstrates the hypergraph composition capability
        # where vision processing feeds into both control and navigation
    
    def test_pipeline_execution(self, middleware_binary):
        """Test execution of module pipelines"""
        result = subprocess.run([str(middleware_binary), "--test"], 
                              capture_output=True, text=True)
        assert result.returncode == 0
        
        # Check pipeline execution
        assert "Executed pipeline with 3 modules" in result.stdout
        
        # Modules should be in ready state after execution (state 2)
        assert "State: 2" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])