#!/usr/bin/env python3
"""
Tests for Enhanced GGUF Integration

This test suite validates the enhanced GGUF functionality including
complete agent state serialization, device configurations, module states,
and P-System membrane support.
"""

import pytest
import subprocess
import tempfile
import os
import json
import time
from pathlib import Path


class TestEnhancedGGUF:
    """Test enhanced GGUF functionality with complete system state"""
    
    @pytest.fixture
    def middleware_binary(self):
        """Path to the robotics middleware binary"""
        return Path(__file__).parent.parent / "buildroot-external" / "package" / "robotics-middleware" / "src" / "robotics-middleware"
    
    @pytest.fixture
    def temp_enhanced_gguf_file(self):
        """Temporary enhanced GGUF file for testing"""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            yield f.name
        os.unlink(f.name)
    
    def test_enhanced_gguf_export_with_membrane(self, middleware_binary, temp_enhanced_gguf_file):
        """Test enhanced GGUF export with P-System membrane"""
        result = subprocess.run([
            str(middleware_binary), 
            "--test", 
            "--export-enhanced", temp_enhanced_gguf_file,
            "--membrane", "test_membrane_v1"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert f"Exported enhanced system state to GGUF: {temp_enhanced_gguf_file}" in result.stdout
        assert "P-System membrane: test_membrane_v1" in result.stdout
        assert "Enhanced GGUF export prepared: 4 tensors, 1 agents, 3 devices, 3 modules" in result.stdout
        assert "Successfully wrote enhanced GGUF file" in result.stdout
        
        # Check file exists and has reasonable size
        assert os.path.exists(temp_enhanced_gguf_file)
        file_size = os.path.getsize(temp_enhanced_gguf_file)
        assert file_size > 16000000  # Should be > 16MB due to tensor data + metadata
    
    def test_enhanced_gguf_import_with_configurations(self, middleware_binary, temp_enhanced_gguf_file):
        """Test enhanced GGUF import with complete system configurations"""
        # First export
        export_result = subprocess.run([
            str(middleware_binary), 
            "--test", 
            "--export-enhanced", temp_enhanced_gguf_file,
            "--membrane", "config_test_membrane"
        ], capture_output=True, text=True)
        assert export_result.returncode == 0
        
        # Then import
        import_result = subprocess.run([
            str(middleware_binary), 
            "--import-enhanced", temp_enhanced_gguf_file
        ], capture_output=True, text=True)
        assert import_result.returncode == 0
        
        # Verify P-System membrane restoration
        assert "Reading P-System membrane: config_test_membrane" in import_result.stdout
        assert "Expected: 4 tensors, 1 agents, 3 devices, 3 modules" in import_result.stdout
        assert "Loaded: 4 tensors, 1 agents, 3 devices, 3 modules" in import_result.stdout
        
        # Verify agent configuration is preserved
        assert "Agent Configurations:" in import_result.stdout
        assert "Agent 0: main_controller (State: 0)" in import_result.stdout
        assert '"cognitive_dimensions":3' in import_result.stdout
        assert '"memory_size":1024' in import_result.stdout
        assert '"attention_heads":8' in import_result.stdout
        assert '"embedding_dim":512' in import_result.stdout
        assert '"scheme_functions":"(define control-loop ...)"' in import_result.stdout
        assert '"neural_symbolic_config":"transformer_agent"' in import_result.stdout
        
        # Verify device configurations are preserved
        assert "Device Configurations:" in import_result.stdout
        assert "Device 0: main_camera (Type: 0, Active: No)" in import_result.stdout
        assert "Device 1: robot_arm (Type: 1, Active: No)" in import_result.stdout
        assert "Device 2: main_imu (Type: 0, Active: No)" in import_result.stdout
        assert '"metadata":"camera_640x480_3ch_30.0fps"' in import_result.stdout
        assert '"metadata":"robot_arm_6dof"' in import_result.stdout
        assert '"metadata":"imu_9axis_100.0Hz"' in import_result.stdout
        
        # Verify module configurations are preserved
        assert "Module Configurations:" in import_result.stdout
        assert "Module 0: vision_processor (Type: 2, State: 2)" in import_result.stdout
        assert "Module 1: arm_controller (Type: 2, State: 2)" in import_result.stdout
        assert "Module 2: navigator (Type: 2, State: 2)" in import_result.stdout
        assert "Description: Computer vision processing module" in import_result.stdout
        assert "Description: Robot control processing module" in import_result.stdout
        assert "Description: Robot navigation processing module" in import_result.stdout
        
        # Verify tensor data is preserved
        assert "Tensor Data:" in import_result.stdout
        assert "Tensor 0: device_0_main_camera" in import_result.stdout
        assert "Tensor 1: device_1_robot_arm" in import_result.stdout
        assert "Tensor 2: device_2_main_imu" in import_result.stdout
        assert "Tensor 3: agent_0_main_controller" in import_result.stdout
        
        # Verify P-System completion message
        assert "Enhanced system state restoration complete." in import_result.stdout
        assert "P-System membrane 'config_test_membrane' contains complete agent kernels and environment tensors." in import_result.stdout
    
    def test_agent_kernels_serialization(self, middleware_binary, temp_enhanced_gguf_file):
        """Test that agent kernels (scheme functions, neural-symbolic configs) are properly serialized"""
        # Export with test data
        export_result = subprocess.run([
            str(middleware_binary), 
            "--test", 
            "--export-enhanced", temp_enhanced_gguf_file,
            "--membrane", "kernel_test"
        ], capture_output=True, text=True)
        assert export_result.returncode == 0
        
        # Import and verify kernels
        import_result = subprocess.run([
            str(middleware_binary), 
            "--import-enhanced", temp_enhanced_gguf_file
        ], capture_output=True, text=True)
        assert import_result.returncode == 0
        
        # Check that agent kernels are preserved
        output = import_result.stdout
        
        # Scheme functions (Lisp code)
        assert '"scheme_functions":"(define control-loop ...)"' in output
        
        # Neural-symbolic configuration
        assert '"neural_symbolic_config":"transformer_agent"' in output
        
        # Agent autonomy state
        assert '"autonomous":false' in output
        
        # Memory and cognitive architecture parameters
        assert '"cognitive_dimensions":3' in output
        assert '"memory_size":1024' in output
        assert '"attention_heads":8' in output
        assert '"embedding_dim":512' in output
    
    def test_device_semantic_specifications(self, middleware_binary, temp_enhanced_gguf_file):
        """Test that device semantic specifications are properly serialized"""
        # Export and import
        subprocess.run([
            str(middleware_binary), 
            "--test", 
            "--export-enhanced", temp_enhanced_gguf_file
        ], capture_output=True, text=True)
        
        import_result = subprocess.run([
            str(middleware_binary), 
            "--import-enhanced", temp_enhanced_gguf_file
        ], capture_output=True, text=True)
        assert import_result.returncode == 0
        
        output = import_result.stdout
        
        # Camera device specifications
        assert '"dimensions":3' in output  # RGB image tensor
        assert '"shape":[480,640,3]' in output  # Height x Width x Channels
        assert '"dtype":0' in output  # UINT8 for image data
        assert '"metadata":"camera_640x480_3ch_30.0fps"' in output
        
        # Robot arm specifications  
        assert '"shape":[6]' in output  # 6-DOF arm
        assert '"dtype":6' in output  # FLOAT32 for joint positions
        assert '"metadata":"robot_arm_6dof"' in output
        
        # IMU specifications
        assert '"shape":[9]' in output  # 9-axis IMU (3 accel + 3 gyro + 3 mag)
        assert '"metadata":"imu_9axis_100.0Hz"' in output
    
    def test_p_system_membrane_structure(self, middleware_binary, temp_enhanced_gguf_file):
        """Test P-System membrane structure and organization"""
        membrane_name = "p_system_test_membrane_v2"
        
        # Export with specific membrane name
        export_result = subprocess.run([
            str(middleware_binary), 
            "--test", 
            "--export-enhanced", temp_enhanced_gguf_file,
            "--membrane", membrane_name
        ], capture_output=True, text=True)
        assert export_result.returncode == 0
        
        # Verify membrane creation
        assert f"Created P-System membrane: {membrane_name}" in export_result.stdout
        assert f"Membrane: {membrane_name}" in export_result.stdout
        
        # Import and verify membrane structure
        import_result = subprocess.run([
            str(middleware_binary), 
            "--import-enhanced", temp_enhanced_gguf_file
        ], capture_output=True, text=True)
        assert import_result.returncode == 0
        
        # Verify membrane is properly read
        assert f"Reading P-System membrane: {membrane_name}" in import_result.stdout
        assert f"P-System membrane '{membrane_name}' contains complete agent kernels and environment tensors." in import_result.stdout
        
        # Verify expected component counts
        assert "Expected: 4 tensors, 1 agents, 3 devices, 3 modules" in import_result.stdout
        assert "Loaded: 4 tensors, 1 agents, 3 devices, 3 modules" in import_result.stdout
    
    def test_environment_tensors_completeness(self, middleware_binary, temp_enhanced_gguf_file):
        """Test that all environment tensors are exported and importable"""
        # Export and import
        subprocess.run([
            str(middleware_binary), 
            "--test", 
            "--export-enhanced", temp_enhanced_gguf_file
        ], capture_output=True, text=True)
        
        import_result = subprocess.run([
            str(middleware_binary), 
            "--import-enhanced", temp_enhanced_gguf_file
        ], capture_output=True, text=True)
        assert import_result.returncode == 0
        
        output = import_result.stdout
        
        # Verify all expected environment tensors are present
        environment_tensors = [
            "device_0_main_camera",    # Visual sensor tensor
            "device_1_robot_arm",      # Actuator state tensor
            "device_2_main_imu",       # Inertial sensor tensor
            "agent_0_main_controller"  # Agent cognitive tensor
        ]
        
        for tensor_name in environment_tensors:
            assert f"Tensor: {tensor_name}" in output
        
        # Verify tensor sizes are preserved
        assert "Size: 921600 bytes" in output  # Camera tensor (480*640*3)
        assert "Size: 24 bytes" in output      # Arm tensor (6 floats)
        assert "Size: 36 bytes" in output      # IMU tensor (9 floats)
        assert "Size: 16777216 bytes" in output # Agent tensor (1024*8*512 floats)
    
    def test_module_dependency_preservation(self, middleware_binary, temp_enhanced_gguf_file):
        """Test that module dependencies and connections are preserved"""
        # Export and import
        subprocess.run([
            str(middleware_binary), 
            "--test", 
            "--export-enhanced", temp_enhanced_gguf_file
        ], capture_output=True, text=True)
        
        import_result = subprocess.run([
            str(middleware_binary), 
            "--import-enhanced", temp_enhanced_gguf_file
        ], capture_output=True, text=True)
        assert import_result.returncode == 0
        
        output = import_result.stdout
        
        # Check that module dependencies are preserved
        # The test creates connections: vision_processor -> arm_controller and vision_processor -> navigator
        # This should show up as dependencies in the vision_processor module
        assert "Dependencies: 1 2" in output  # Module IDs 1 and 2 depend on module 0


class TestGGUFCompatibility:
    """Test compatibility with standard GGUF and enhanced formats"""
    
    @pytest.fixture
    def middleware_binary(self):
        """Path to the robotics middleware binary"""
        return Path(__file__).parent.parent / "buildroot-external" / "package" / "robotics-middleware" / "src" / "robotics-middleware"
    
    def test_backwards_compatibility(self, middleware_binary):
        """Test that standard GGUF export/import still works alongside enhanced version"""
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            standard_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            enhanced_file = f.name
        
        try:
            # Test standard GGUF export
            std_export = subprocess.run([
                str(middleware_binary), 
                "--test", 
                "--export", standard_file
            ], capture_output=True, text=True)
            assert std_export.returncode == 0
            assert f"Exported system state to GGUF: {standard_file}" in std_export.stdout
            
            # Test enhanced GGUF export
            enh_export = subprocess.run([
                str(middleware_binary), 
                "--test", 
                "--export-enhanced", enhanced_file
            ], capture_output=True, text=True)
            assert enh_export.returncode == 0
            assert f"Exported enhanced system state to GGUF: {enhanced_file}" in enh_export.stdout
            
            # Both files should exist with different sizes (enhanced should be larger)
            assert os.path.exists(standard_file)
            assert os.path.exists(enhanced_file)
            
            std_size = os.path.getsize(standard_file)
            enh_size = os.path.getsize(enhanced_file)
            assert enh_size > std_size  # Enhanced file should be larger due to metadata
            
            # Test standard import
            std_import = subprocess.run([
                str(middleware_binary), 
                "--import", standard_file
            ], capture_output=True, text=True)
            assert std_import.returncode == 0
            assert f"Imported system state from GGUF: {standard_file}" in std_import.stdout
            
            # Test enhanced import
            enh_import = subprocess.run([
                str(middleware_binary), 
                "--import-enhanced", enhanced_file
            ], capture_output=True, text=True)
            assert enh_import.returncode == 0
            assert f"Imported enhanced system state from GGUF: {enhanced_file}" in enh_import.stdout
            
        finally:
            if os.path.exists(standard_file):
                os.unlink(standard_file)
            if os.path.exists(enhanced_file):
                os.unlink(enhanced_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])