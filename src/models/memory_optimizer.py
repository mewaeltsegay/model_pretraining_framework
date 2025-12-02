"""
Memory optimization utilities for efficient training on RTX 4050 6GB GPU.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, Tuple, List
import logging
import gc
import psutil
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Comprehensive memory optimization for GPU-constrained training.
    
    This class provides:
    - Automatic memory monitoring and alerts
    - Dynamic batch size adjustment based on available memory
    - Mixed precision training setup and management
    - Memory cleanup and garbage collection utilities
    - GPU memory usage reporting and optimization suggestions
    """
    
    def __init__(self, max_gpu_memory_gb: float = 5.5, enable_mixed_precision: bool = True):
        """
        Initialize MemoryOptimizer.
        
        Args:
            max_gpu_memory_gb: Maximum GPU memory to use in GB
            enable_mixed_precision: Whether to enable mixed precision training
        """
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.enable_mixed_precision = enable_mixed_precision
        
        # Mixed precision components
        self.scaler: Optional[GradScaler] = None
        if enable_mixed_precision and torch.cuda.is_available():
            try:
                # Use new API if available (PyTorch 2.1+)
                from torch.amp import GradScaler as NewGradScaler
                self.scaler = NewGradScaler('cuda')
            except ImportError:
                # Fallback to old API
                self.scaler = GradScaler()
        
        # Memory tracking
        self.memory_history: List[Dict[str, float]] = []
        self.peak_memory_usage = 0.0
        self.oom_count = 0
        
        # Dynamic batch sizing
        self.original_batch_size: Optional[int] = None
        self.current_batch_size: Optional[int] = None
        self.min_batch_size = 1
        
        logger.info(f"MemoryOptimizer initialized with {max_gpu_memory_gb}GB limit")
    
    def setup_mixed_precision(self) -> GradScaler:
        """
        Set up mixed precision training components.
        
        Returns:
            GradScaler instance for mixed precision training
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, mixed precision disabled")
            return None
        
        if not self.enable_mixed_precision:
            logger.info("Mixed precision disabled by configuration")
            return None
        
        if self.scaler is None:
            try:
                # Use new API if available (PyTorch 2.1+)
                from torch.amp import GradScaler as NewGradScaler
                self.scaler = NewGradScaler('cuda')
            except ImportError:
                # Fallback to old API
                self.scaler = GradScaler()
        
        logger.info("Mixed precision training enabled with GradScaler")
        return self.scaler
    
    @contextmanager
    def autocast_context(self):
        """
        Context manager for mixed precision forward pass.
        
        Usage:
            with optimizer.autocast_context():
                outputs = model(inputs)
        """
        if self.enable_mixed_precision and torch.cuda.is_available():
            with autocast():
                yield
        else:
            yield
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU and system memory usage.
        
        Returns:
            Dictionary with memory statistics in GB
        """
        memory_stats = {}
        
        if torch.cuda.is_available():
            # GPU memory
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            
            device_props = torch.cuda.get_device_properties(0)
            total_gpu = device_props.total_memory / (1024**3)
            
            memory_stats.update({
                'gpu_allocated': allocated,
                'gpu_reserved': reserved,
                'gpu_max_allocated': max_allocated,
                'gpu_total': total_gpu,
                'gpu_free': total_gpu - allocated,
                'gpu_utilization_percent': (allocated / total_gpu) * 100
            })
            
            # Update peak usage
            self.peak_memory_usage = max(self.peak_memory_usage, allocated)
        
        # System RAM
        ram = psutil.virtual_memory()
        memory_stats.update({
            'system_ram_total': ram.total / (1024**3),
            'system_ram_used': ram.used / (1024**3),
            'system_ram_percent': ram.percent
        })
        
        return memory_stats
    
    def monitor_memory(self, log_stats: bool = False) -> Dict[str, float]:
        """
        Monitor memory usage and add to history.
        
        Args:
            log_stats: Whether to log memory statistics
            
        Returns:
            Current memory statistics
        """
        stats = self.get_current_memory_usage()
        stats['timestamp'] = time.time()
        
        # Add to history (keep last 100 entries)
        self.memory_history.append(stats)
        if len(self.memory_history) > 100:
            self.memory_history.pop(0)
        
        if log_stats and torch.cuda.is_available():
            logger.info(
                f"GPU Memory: {stats['gpu_allocated']:.2f}GB / {stats['gpu_total']:.2f}GB "
                f"({stats['gpu_utilization_percent']:.1f}%)"
            )
        
        return stats
    
    def check_memory_constraints(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Check if current memory usage is within constraints.
        
        Returns:
            Tuple of (is_within_limits, message, suggestions)
        """
        stats = self.get_current_memory_usage()
        suggestions = {}
        
        if not torch.cuda.is_available():
            return True, "No GPU available", {}
        
        gpu_used = stats['gpu_allocated']
        gpu_percent = stats['gpu_utilization_percent']
        
        # Check hard limit
        if gpu_used > self.max_gpu_memory_gb:
            return False, f"GPU memory ({gpu_used:.2f}GB) exceeds limit ({self.max_gpu_memory_gb}GB)", {
                'reduce_batch_size': True,
                'enable_gradient_checkpointing': True,
                'use_mixed_precision': True
            }
        
        # Check soft limits and provide suggestions
        if gpu_percent > 90:
            suggestions['critical'] = "Memory usage critically high"
            suggestions['reduce_batch_size'] = True
        elif gpu_percent > 80:
            suggestions['warning'] = "Memory usage high, consider optimizations"
            suggestions['enable_gradient_accumulation'] = True
        elif gpu_percent > 70:
            suggestions['info'] = "Memory usage moderate, room for optimization"
        
        message = f"Memory usage: {gpu_used:.2f}GB ({gpu_percent:.1f}%)"
        return True, message, suggestions
    
    def optimize_memory_automatically(self, model: nn.Module) -> Dict[str, Any]:
        """
        Apply automatic memory optimizations based on current usage.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Dictionary of applied optimizations
        """
        optimizations = {}
        
        # Clear cache first
        self.clear_memory_cache()
        optimizations['cache_cleared'] = True
        
        # Check current memory state
        within_limits, message, suggestions = self.check_memory_constraints()
        
        if not within_limits or suggestions.get('critical'):
            # Apply aggressive optimizations
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                optimizations['gradient_checkpointing'] = True
            
            # Enable mixed precision if not already enabled
            if not self.enable_mixed_precision and torch.cuda.is_available():
                self.enable_mixed_precision = True
                try:
                    # Use new API if available (PyTorch 2.1+)
                    from torch.amp import GradScaler as NewGradScaler
                    self.scaler = NewGradScaler('cuda')
                except ImportError:
                    # Fallback to old API
                    self.scaler = GradScaler()
                optimizations['mixed_precision_enabled'] = True
        
        # Log optimizations
        if optimizations:
            logger.info(f"Applied automatic optimizations: {optimizations}")
        
        return optimizations
    
    def suggest_batch_size(self, current_batch_size: int, sequence_length: int) -> int:
        """
        Suggest optimal batch size based on available memory.
        
        Args:
            current_batch_size: Current batch size
            sequence_length: Sequence length
            
        Returns:
            Suggested batch size
        """
        if not torch.cuda.is_available():
            return current_batch_size
        
        stats = self.get_current_memory_usage()
        available_memory = stats['gpu_free']
        
        # Rough estimation: each token uses ~4 bytes in FP32, ~2 bytes in FP16
        bytes_per_token = 2 if self.enable_mixed_precision else 4
        
        # Estimate memory per sample (very rough approximation)
        memory_per_sample = (sequence_length * bytes_per_token * 4) / (1024**3)  # Factor of 4 for gradients, activations
        
        # Calculate maximum batch size that fits in available memory
        max_batch_size = max(1, int(available_memory * 0.8 / memory_per_sample))  # Use 80% of available
        
        suggested_batch_size = min(current_batch_size, max_batch_size)
        
        if suggested_batch_size != current_batch_size:
            logger.info(
                f"Suggested batch size: {suggested_batch_size} (current: {current_batch_size}, "
                f"available memory: {available_memory:.2f}GB)"
            )
        
        return suggested_batch_size
    
    def handle_oom_error(self, current_batch_size: int) -> int:
        """
        Handle CUDA Out of Memory error by reducing batch size.
        
        Args:
            current_batch_size: Current batch size that caused OOM
            
        Returns:
            New reduced batch size
        """
        self.oom_count += 1
        
        # Clear memory cache
        self.clear_memory_cache()
        
        # Reduce batch size by half, but not below minimum
        new_batch_size = max(self.min_batch_size, current_batch_size // 2)
        
        logger.warning(
            f"CUDA OOM error #{self.oom_count}. Reducing batch size from {current_batch_size} to {new_batch_size}"
        )
        
        return new_batch_size
    
    def clear_memory_cache(self) -> None:
        """Clear GPU memory cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        gc.collect()
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory usage report.
        
        Returns:
            Dictionary containing detailed memory statistics and history
        """
        current_stats = self.get_current_memory_usage()
        
        report = {
            'current_usage': current_stats,
            'peak_gpu_usage_gb': self.peak_memory_usage,
            'oom_count': self.oom_count,
            'mixed_precision_enabled': self.enable_mixed_precision,
            'max_gpu_memory_limit_gb': self.max_gpu_memory_gb,
        }
        
        # Add history statistics if available
        if self.memory_history:
            gpu_usage_history = [entry.get('gpu_allocated', 0) for entry in self.memory_history]
            report['memory_history'] = {
                'samples': len(gpu_usage_history),
                'avg_gpu_usage_gb': sum(gpu_usage_history) / len(gpu_usage_history),
                'min_gpu_usage_gb': min(gpu_usage_history),
                'max_gpu_usage_gb': max(gpu_usage_history),
            }
        
        # Add optimization suggestions
        within_limits, message, suggestions = self.check_memory_constraints()
        report['status'] = {
            'within_limits': within_limits,
            'message': message,
            'suggestions': suggestions
        }
        
        return report
    
    def log_memory_summary(self) -> None:
        """Log a summary of memory usage and optimizations."""
        report = self.get_memory_report()
        
        logger.info("=== Memory Usage Summary ===")
        if torch.cuda.is_available():
            current = report['current_usage']
            logger.info(f"Current GPU Usage: {current['gpu_allocated']:.2f}GB / {current['gpu_total']:.2f}GB")
            logger.info(f"Peak GPU Usage: {report['peak_gpu_usage_gb']:.2f}GB")
            logger.info(f"GPU Utilization: {current['gpu_utilization_percent']:.1f}%")
        
        logger.info(f"OOM Errors: {report['oom_count']}")
        logger.info(f"Mixed Precision: {'Enabled' if report['mixed_precision_enabled'] else 'Disabled'}")
        
        # Log suggestions if any
        suggestions = report['status']['suggestions']
        if suggestions:
            logger.info(f"Optimization Suggestions: {suggestions}")
        
        logger.info("=== End Memory Summary ===")


class DynamicBatchSizer:
    """
    Dynamically adjusts batch size based on available GPU memory.
    """
    
    def __init__(self, initial_batch_size: int, min_batch_size: int = 1, max_batch_size: int = 32):
        """
        Initialize DynamicBatchSizer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
        """
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        
        self.oom_history = []
        self.success_history = []
        
    def adjust_for_oom(self) -> int:
        """
        Adjust batch size after OOM error.
        
        Returns:
            New batch size
        """
        self.oom_history.append(self.current_batch_size)
        
        # Reduce by half, but not below minimum
        self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        
        logger.info(f"Batch size reduced to {self.current_batch_size} due to OOM")
        return self.current_batch_size
    
    def record_success(self) -> None:
        """Record successful training step with current batch size."""
        self.success_history.append(self.current_batch_size)
        
        # Keep only recent history
        if len(self.success_history) > 10:
            self.success_history.pop(0)
    
    def can_increase_batch_size(self, memory_utilization: float) -> bool:
        """
        Check if batch size can be safely increased.
        
        Args:
            memory_utilization: Current GPU memory utilization (0-100)
            
        Returns:
            True if batch size can be increased
        """
        # Only increase if memory utilization is low and we have successful history
        return (
            memory_utilization < 60 and  # Low memory usage
            len(self.success_history) >= 5 and  # Stable recent history
            self.current_batch_size < self.max_batch_size and  # Not at maximum
            self.current_batch_size not in self.oom_history[-3:]  # Haven't had recent OOM at this size
        )
    
    def increase_batch_size(self) -> int:
        """
        Cautiously increase batch size.
        
        Returns:
            New batch size
        """
        old_size = self.current_batch_size
        self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
        
        if self.current_batch_size != old_size:
            logger.info(f"Batch size increased to {self.current_batch_size}")
        
        return self.current_batch_size
    
    def get_current_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size
    
    def reset(self) -> None:
        """Reset to initial batch size."""
        self.current_batch_size = self.initial_batch_size
        self.oom_history.clear()
        self.success_history.clear()