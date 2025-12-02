"""
Hardware validation and constraint checking for Qwen pretraining.

This module provides comprehensive validation of hardware constraints,
automatic parameter adjustment for hardware limitations, and pre-training
system checks with warnings.
"""

import logging
import platform
import psutil
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np

from ..config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class HardwareConstraints:
    """Hardware constraint specifications for RTX 4050 6GB."""
    
    # GPU constraints
    min_gpu_memory_gb: float = 4.0
    recommended_gpu_memory_gb: float = 6.0
    max_gpu_memory_gb: float = 6.0
    
    # Memory usage thresholds
    safe_memory_usage_percent: float = 85.0  # Leave 15% buffer
    critical_memory_usage_percent: float = 95.0
    
    # Model constraints
    max_model_parameters: int = 1_000_000_000  # 1B parameters max
    max_sequence_length: int = 2048
    min_batch_size: int = 1
    max_batch_size: int = 32
    
    # System constraints
    min_ram_gb: float = 8.0
    recommended_ram_gb: float = 16.0
    min_disk_space_gb: float = 10.0


@dataclass
class ValidationResult:
    """Result of hardware validation."""
    
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    suggestions: List[str]
    adjusted_config: Optional[TrainingConfig] = None


class HardwareValidator:
    """
    Comprehensive hardware validator for Qwen pretraining.
    
    Validates training parameters against RTX 4050 6GB constraints,
    implements automatic parameter adjustment for hardware limitations,
    and provides pre-training system checks with warnings.
    """
    
    def __init__(self, constraints: Optional[HardwareConstraints] = None):
        """
        Initialize hardware validator.
        
        Args:
            constraints: Hardware constraints specification
        """
        self.constraints = constraints or HardwareConstraints()
        self.system_info = self._gather_system_info()
        
        logger.info("HardwareValidator initialized")
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information."""
        
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_free_gb": psutil.disk_usage('.').free / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_info": []
        }
        
        # Gather GPU information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "major": props.major,
                    "minor": props.minor,
                    "multi_processor_count": props.multi_processor_count
                }
                info["gpu_info"].append(gpu_info)
        
        return info
    
    def validate_system_requirements(self) -> ValidationResult:
        """
        Validate basic system requirements.
        
        Returns:
            ValidationResult with system validation status
        """
        warnings = []
        errors = []
        suggestions = []
        
        # Check CUDA availability
        if not self.system_info["cuda_available"]:
            errors.append("CUDA is not available. GPU training is required.")
            suggestions.append("Install CUDA-compatible PyTorch version")
        
        # Check GPU availability
        if self.system_info["gpu_count"] == 0:
            errors.append("No GPU detected. GPU training is required.")
        
        # Check GPU memory
        if self.system_info["gpu_info"]:
            gpu_memory = self.system_info["gpu_info"][0]["total_memory_gb"]
            
            if gpu_memory < self.constraints.min_gpu_memory_gb:
                errors.append(
                    f"GPU memory ({gpu_memory:.1f}GB) is below minimum requirement "
                    f"({self.constraints.min_gpu_memory_gb}GB)"
                )
            elif gpu_memory < self.constraints.recommended_gpu_memory_gb:
                warnings.append(
                    f"GPU memory ({gpu_memory:.1f}GB) is below recommended "
                    f"({self.constraints.recommended_gpu_memory_gb}GB)"
                )
                suggestions.append("Consider reducing batch size or sequence length")
        
        # Check RAM
        ram_gb = self.system_info["ram_total_gb"]
        if ram_gb < self.constraints.min_ram_gb:
            warnings.append(
                f"System RAM ({ram_gb:.1f}GB) is below recommended "
                f"({self.constraints.min_ram_gb}GB)"
            )
            suggestions.append("Close other applications to free up RAM")
        
        # Check disk space
        disk_gb = self.system_info["disk_free_gb"]
        if disk_gb < self.constraints.min_disk_space_gb:
            warnings.append(
                f"Available disk space ({disk_gb:.1f}GB) is low. "
                f"Recommended: {self.constraints.min_disk_space_gb}GB"
            )
            suggestions.append("Free up disk space for checkpoints and logs")
        
        # Check PyTorch version compatibility
        pytorch_version = self.system_info["pytorch_version"]
        if not pytorch_version.startswith(("1.13", "2.0", "2.1", "2.2", "2.3")):
            warnings.append(f"PyTorch version {pytorch_version} may not be fully compatible")
            suggestions.append("Consider using PyTorch 2.0+ for best performance")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            suggestions=suggestions
        )
    
    def validate_training_config(self, config: TrainingConfig) -> ValidationResult:
        """
        Validate training configuration against hardware constraints.
        
        Args:
            config: Training configuration to validate
            
        Returns:
            ValidationResult with configuration validation status
        """
        warnings = []
        errors = []
        suggestions = []
        
        # Validate batch size
        if config.batch_size < self.constraints.min_batch_size:
            errors.append(f"Batch size ({config.batch_size}) is below minimum ({self.constraints.min_batch_size})")
        elif config.batch_size > self.constraints.max_batch_size:
            warnings.append(f"Batch size ({config.batch_size}) is quite large for RTX 4050")
            suggestions.append("Consider reducing batch size to avoid OOM errors")
        
        # Validate sequence length
        if config.max_sequence_length > self.constraints.max_sequence_length:
            warnings.append(
                f"Sequence length ({config.max_sequence_length}) is very long. "
                f"Recommended max: {self.constraints.max_sequence_length}"
            )
            suggestions.append("Consider reducing sequence length for better memory efficiency")
        
        # Validate GPU memory configuration
        if self.system_info["gpu_info"]:
            gpu_memory = self.system_info["gpu_info"][0]["total_memory_gb"]
            
            if config.max_gpu_memory_gb > gpu_memory:
                errors.append(
                    f"Configured max GPU memory ({config.max_gpu_memory_gb}GB) "
                    f"exceeds available GPU memory ({gpu_memory:.1f}GB)"
                )
                suggestions.append(f"Set max_gpu_memory_gb to {gpu_memory * 0.9:.1f}")
            
            # Estimate memory usage
            memory_estimate = self._estimate_memory_usage(config)
            if memory_estimate > config.max_gpu_memory_gb:
                warnings.append(
                    f"Estimated memory usage ({memory_estimate:.2f}GB) may exceed "
                    f"configured limit ({config.max_gpu_memory_gb}GB)"
                )
                suggestions.append("Consider reducing batch size or sequence length")
        
        # Validate gradient accumulation
        effective_batch_size = config.get_effective_batch_size()
        if effective_batch_size > 64:
            warnings.append(f"Effective batch size ({effective_batch_size}) is very large")
            suggestions.append("Large effective batch sizes may slow convergence")
        
        # Validate learning rate
        if config.learning_rate > 1e-3:
            warnings.append(f"Learning rate ({config.learning_rate}) is quite high")
            suggestions.append("High learning rates may cause training instability")
        elif config.learning_rate < 1e-6:
            warnings.append(f"Learning rate ({config.learning_rate}) is very low")
            suggestions.append("Very low learning rates may slow training")
        
        # Validate checkpoint settings
        if config.save_steps < 100:
            warnings.append("Very frequent checkpoint saving may slow training")
        elif config.save_steps > 5000:
            warnings.append("Infrequent checkpoints increase risk of losing progress")
        
        # Check file paths
        if not Path(config.tokenizer_path).exists():
            errors.append(f"Tokenizer path does not exist: {config.tokenizer_path}")
        
        if not Path(config.data_dir).exists():
            errors.append(f"Data directory does not exist: {config.data_dir}")
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            suggestions=suggestions
        )
    
    def _estimate_memory_usage(self, config: TrainingConfig) -> float:
        """
        Estimate GPU memory usage for the given configuration.
        
        Args:
            config: Training configuration
            
        Returns:
            Estimated memory usage in GB
        """
        # Rough estimation based on model size and batch configuration
        # This is a simplified estimation - actual usage may vary
        
        # Base model memory (Qwen-0.5B in FP16)
        model_params = 500_000_000  # 500M parameters
        bytes_per_param = 2 if config.use_mixed_precision else 4  # FP16 vs FP32
        model_memory = model_params * bytes_per_param / (1024**3)  # GB
        
        # Optimizer memory (AdamW stores 2 copies of parameters)
        optimizer_memory = model_memory * 2
        
        # Activation memory (depends on batch size and sequence length)
        hidden_size = 1024  # Qwen-0.5B hidden size
        num_layers = 24     # Qwen-0.5B layers
        
        activation_memory_per_token = hidden_size * num_layers * bytes_per_param / (1024**3)
        total_tokens = config.batch_size * config.max_sequence_length
        activation_memory = activation_memory_per_token * total_tokens
        
        # Gradient memory
        gradient_memory = model_memory
        
        # Buffer and overhead (20% of total)
        base_memory = model_memory + optimizer_memory + activation_memory + gradient_memory
        overhead = base_memory * 0.2
        
        total_memory = base_memory + overhead
        
        return total_memory
    
    def auto_adjust_config(self, config: TrainingConfig) -> Tuple[TrainingConfig, List[str]]:
        """
        Automatically adjust configuration for hardware constraints.
        
        Args:
            config: Original training configuration
            
        Returns:
            Tuple of (adjusted_config, list_of_adjustments_made)
        """
        adjusted_config = TrainingConfig(**config.to_dict())  # Create copy
        adjustments = []
        
        if not self.system_info["gpu_info"]:
            return adjusted_config, adjustments
        
        gpu_memory = self.system_info["gpu_info"][0]["total_memory_gb"]
        
        # Adjust max GPU memory to safe limit
        safe_gpu_memory = gpu_memory * (self.constraints.safe_memory_usage_percent / 100)
        if adjusted_config.max_gpu_memory_gb > safe_gpu_memory:
            adjusted_config.max_gpu_memory_gb = safe_gpu_memory
            adjustments.append(f"Reduced max_gpu_memory_gb to {safe_gpu_memory:.1f}GB")
        
        # Iteratively adjust batch size and sequence length
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            memory_estimate = self._estimate_memory_usage(adjusted_config)
            
            if memory_estimate <= adjusted_config.max_gpu_memory_gb:
                break
            
            # Try reducing batch size first
            if adjusted_config.batch_size > 1:
                old_batch_size = adjusted_config.batch_size
                adjusted_config.batch_size = max(1, adjusted_config.batch_size // 2)
                adjusted_config.gradient_accumulation_steps *= 2  # Maintain effective batch size
                adjustments.append(
                    f"Reduced batch_size from {old_batch_size} to {adjusted_config.batch_size}, "
                    f"increased gradient_accumulation_steps to {adjusted_config.gradient_accumulation_steps}"
                )
            
            # If batch size is already 1, reduce sequence length
            elif adjusted_config.max_sequence_length > 128:
                old_seq_len = adjusted_config.max_sequence_length
                adjusted_config.max_sequence_length = max(128, adjusted_config.max_sequence_length // 2)
                adjustments.append(
                    f"Reduced max_sequence_length from {old_seq_len} to {adjusted_config.max_sequence_length}"
                )
            
            else:
                # Can't reduce further
                adjustments.append("Warning: Could not reduce memory usage further")
                break
            
            iteration += 1
        
        # Ensure gradient checkpointing is enabled for memory efficiency
        if not adjusted_config.gradient_checkpointing:
            adjusted_config.gradient_checkpointing = True
            adjustments.append("Enabled gradient_checkpointing for memory efficiency")
        
        # Ensure mixed precision is enabled
        if not adjusted_config.use_mixed_precision:
            adjusted_config.use_mixed_precision = True
            adjustments.append("Enabled mixed_precision for memory efficiency")
        
        # Adjust dataloader workers for Windows compatibility
        if platform.system() == "Windows" and adjusted_config.dataloader_num_workers > 0:
            adjusted_config.dataloader_num_workers = 0
            adjustments.append("Set dataloader_num_workers to 0 for Windows compatibility")
        
        return adjusted_config, adjustments
    
    def run_pre_training_checks(self, config: TrainingConfig) -> ValidationResult:
        """
        Run comprehensive pre-training system checks.
        
        Args:
            config: Training configuration
            
        Returns:
            ValidationResult with comprehensive check results
        """
        all_warnings = []
        all_errors = []
        all_suggestions = []
        
        # System requirements check
        system_result = self.validate_system_requirements()
        all_warnings.extend(system_result.warnings)
        all_errors.extend(system_result.errors)
        all_suggestions.extend(system_result.suggestions)
        
        # Configuration validation
        config_result = self.validate_training_config(config)
        all_warnings.extend(config_result.warnings)
        all_errors.extend(config_result.errors)
        all_suggestions.extend(config_result.suggestions)
        
        # Additional runtime checks
        runtime_warnings, runtime_errors, runtime_suggestions = self._run_runtime_checks(config)
        all_warnings.extend(runtime_warnings)
        all_errors.extend(runtime_errors)
        all_suggestions.extend(runtime_suggestions)
        
        # Auto-adjust configuration if there are issues
        adjusted_config = None
        if all_warnings or all_errors:
            adjusted_config, adjustments = self.auto_adjust_config(config)
            if adjustments:
                all_suggestions.extend([f"Auto-adjustment: {adj}" for adj in adjustments])
        
        is_valid = len(all_errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            warnings=all_warnings,
            errors=all_errors,
            suggestions=all_suggestions,
            adjusted_config=adjusted_config
        )
    
    def _run_runtime_checks(self, config: TrainingConfig) -> Tuple[List[str], List[str], List[str]]:
        """Run additional runtime checks."""
        warnings = []
        errors = []
        suggestions = []
        
        # Check current GPU memory usage
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                current_memory = torch.cuda.memory_allocated() / (1024**3)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if current_memory > 0.1:  # More than 100MB already allocated
                    warnings.append(f"GPU memory already in use: {current_memory:.2f}GB")
                    suggestions.append("Consider restarting Python to free GPU memory")
                
                # Check available memory vs requirements
                available_memory = total_memory - current_memory
                estimated_usage = self._estimate_memory_usage(config)
                
                if estimated_usage > available_memory:
                    errors.append(
                        f"Estimated memory usage ({estimated_usage:.2f}GB) exceeds "
                        f"available memory ({available_memory:.2f}GB)"
                    )
                    suggestions.append("Reduce batch size or sequence length")
                
            except Exception as e:
                warnings.append(f"Could not check GPU memory status: {e}")
        
        # Check system load
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            warnings.append(f"High CPU usage detected: {cpu_percent:.1f}%")
            suggestions.append("Close other applications to reduce system load")
        
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 80:
            warnings.append(f"High RAM usage detected: {memory_percent:.1f}%")
            suggestions.append("Close other applications to free RAM")
        
        # Check disk space for checkpoints
        try:
            output_dir = Path(config.output_dir)
            if output_dir.exists():
                disk_usage = psutil.disk_usage(str(output_dir))
                free_gb = disk_usage.free / (1024**3)
                
                # Estimate checkpoint size (rough estimate)
                estimated_checkpoint_size = self._estimate_memory_usage(config) * 3  # Model + optimizer + scheduler
                required_space = estimated_checkpoint_size * config.save_total_limit * 2  # Safety margin
                
                if free_gb < required_space:
                    warnings.append(
                        f"Low disk space: {free_gb:.1f}GB available, "
                        f"estimated need: {required_space:.1f}GB"
                    )
                    suggestions.append("Free up disk space or reduce save_total_limit")
        
        except Exception as e:
            warnings.append(f"Could not check disk space: {e}")
        
        return warnings, errors, suggestions
    
    def get_system_report(self) -> Dict[str, Any]:
        """
        Get comprehensive system report.
        
        Returns:
            Dictionary containing detailed system information
        """
        report = {
            "system_info": self.system_info,
            "constraints": {
                "min_gpu_memory_gb": self.constraints.min_gpu_memory_gb,
                "recommended_gpu_memory_gb": self.constraints.recommended_gpu_memory_gb,
                "max_gpu_memory_gb": self.constraints.max_gpu_memory_gb,
                "safe_memory_usage_percent": self.constraints.safe_memory_usage_percent,
                "max_model_parameters": self.constraints.max_model_parameters,
                "max_sequence_length": self.constraints.max_sequence_length,
                "min_ram_gb": self.constraints.min_ram_gb,
                "recommended_ram_gb": self.constraints.recommended_ram_gb
            },
            "recommendations": []
        }
        
        # Add recommendations based on system
        if self.system_info["gpu_info"]:
            gpu_memory = self.system_info["gpu_info"][0]["total_memory_gb"]
            if gpu_memory <= 6:
                report["recommendations"].extend([
                    "Use batch_size=1 or 2 for 6GB GPU",
                    "Enable gradient_checkpointing and mixed_precision",
                    "Consider max_sequence_length=512 or lower",
                    "Use gradient_accumulation_steps to simulate larger batches"
                ])
            elif gpu_memory <= 8:
                report["recommendations"].extend([
                    "Use batch_size=2-4 for 8GB GPU",
                    "Enable gradient_checkpointing for larger models",
                    "Consider max_sequence_length=1024"
                ])
        
        return report
    
    def print_validation_report(self, result: ValidationResult) -> None:
        """
        Print a formatted validation report.
        
        Args:
            result: ValidationResult to print
        """
        print("\n" + "=" * 60)
        print("HARDWARE VALIDATION REPORT")
        print("=" * 60)
        
        if result.is_valid:
            print("✅ VALIDATION PASSED")
        else:
            print("❌ VALIDATION FAILED")
        
        if result.errors:
            print(f"\n🚨 ERRORS ({len(result.errors)}):")
            for i, error in enumerate(result.errors, 1):
                print(f"  {i}. {error}")
        
        if result.warnings:
            print(f"\n⚠️  WARNINGS ({len(result.warnings)}):")
            for i, warning in enumerate(result.warnings, 1):
                print(f"  {i}. {warning}")
        
        if result.suggestions:
            print(f"\n💡 SUGGESTIONS ({len(result.suggestions)}):")
            for i, suggestion in enumerate(result.suggestions, 1):
                print(f"  {i}. {suggestion}")
        
        if result.adjusted_config:
            print(f"\n🔧 AUTO-ADJUSTED CONFIGURATION AVAILABLE")
            print("   Use the adjusted configuration to resolve issues")
        
        print("=" * 60)