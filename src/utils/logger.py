"""
Comprehensive logging system for Qwen pretraining with structured logging,
GPU memory monitoring, and training progress tracking.
"""
import logging
import json
import time
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import psutil
import torch
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Logging levels for different verbosity."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class TrainingMetrics:
    """Structured training metrics."""
    step: int
    epoch: int
    loss: float
    learning_rate: float
    gpu_memory_allocated: float
    gpu_memory_free: float
    cpu_percent: float
    memory_percent: float
    timestamp: str
    batch_size: Optional[int] = None
    gradient_norm: Optional[float] = None
    tokens_per_second: Optional[float] = None


@dataclass
class ValidationMetrics:
    """Structured validation metrics."""
    step: int
    epoch: int
    val_loss: float
    val_perplexity: float
    is_best: bool
    timestamp: str
    num_batches: int
    validation_time: float


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: str
    gpu_memory_allocated: float
    gpu_memory_reserved: float
    gpu_memory_free: float
    gpu_memory_total: float
    gpu_utilization: Optional[float]
    gpu_temperature: Optional[float]
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float


class ProgressTracker:
    """Training progress tracker with ETA calculation."""
    
    def __init__(self, total_steps: int, total_epochs: int):
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of training steps
            total_epochs: Total number of epochs
        """
        self.total_steps = total_steps
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.step_times: List[float] = []
        self.epoch_times: List[float] = []
        self.last_step_time = self.start_time
        self.last_epoch_time = self.start_time
        
    def update_step(self, current_step: int) -> Dict[str, Any]:
        """
        Update step progress and calculate ETA.
        
        Args:
            current_step: Current training step
            
        Returns:
            Progress information including ETA
        """
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time
        
        # Keep only recent step times for better ETA estimation
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]
        
        # Calculate progress and ETA
        progress_percent = (current_step / self.total_steps) * 100
        avg_step_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - current_step
        eta_seconds = remaining_steps * avg_step_time
        
        return {
            "current_step": current_step,
            "total_steps": self.total_steps,
            "progress_percent": progress_percent,
            "avg_step_time": avg_step_time,
            "eta_seconds": eta_seconds,
            "eta_formatted": str(timedelta(seconds=int(eta_seconds))),
            "elapsed_time": current_time - self.start_time,
            "steps_per_second": 1.0 / avg_step_time if avg_step_time > 0 else 0
        }
    
    def update_epoch(self, current_epoch: int) -> Dict[str, Any]:
        """
        Update epoch progress.
        
        Args:
            current_epoch: Current epoch number
            
        Returns:
            Epoch progress information
        """
        current_time = time.time()
        epoch_time = current_time - self.last_epoch_time
        self.epoch_times.append(epoch_time)
        self.last_epoch_time = current_time
        
        # Keep only recent epoch times
        if len(self.epoch_times) > 10:
            self.epoch_times = self.epoch_times[-10:]
        
        progress_percent = (current_epoch / self.total_epochs) * 100
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - current_epoch
        eta_seconds = remaining_epochs * avg_epoch_time
        
        return {
            "current_epoch": current_epoch,
            "total_epochs": self.total_epochs,
            "progress_percent": progress_percent,
            "avg_epoch_time": avg_epoch_time,
            "eta_seconds": eta_seconds,
            "eta_formatted": str(timedelta(seconds=int(eta_seconds)))
        }


class GPUMonitor:
    """GPU monitoring utilities."""
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        if not torch.cuda.is_available():
            return {
                "available": False,
                "device_count": 0,
                "current_device": None
            }
        
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(current_device)
        
        return {
            "available": True,
            "device_count": device_count,
            "current_device": current_device,
            "device_name": device_props.name,
            "total_memory": device_props.total_memory / (1024**3),  # GB
            "major": device_props.major,
            "minor": device_props.minor,
            "multi_processor_count": device_props.multi_processor_count
        }
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}
        
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        
        device_props = torch.cuda.get_device_properties(0)
        total = device_props.total_memory / (1024**3)  # GB
        free = total - reserved
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": total,
            "utilization_percent": (allocated / total) * 100 if total > 0 else 0
        }
    
    @staticmethod
    def get_gpu_utilization() -> Optional[float]:
        """Get GPU utilization percentage (requires nvidia-ml-py)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return utilization.gpu
        except ImportError:
            return None
        except Exception:
            return None
    
    @staticmethod
    def get_gpu_temperature() -> Optional[float]:
        """Get GPU temperature (requires nvidia-ml-py)."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            return temp
        except ImportError:
            return None
        except Exception:
            return None


class StructuredLogger:
    """
    Structured logger with JSON output, GPU monitoring, and progress tracking.
    """
    
    def __init__(self, 
                 name: str,
                 log_dir: str,
                 log_level: LogLevel = LogLevel.INFO,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 enable_json: bool = True):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level
            enable_console: Enable console output
            enable_file: Enable file output
            enable_json: Enable JSON structured logging
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.value))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if enable_file:
            file_handler = logging.FileHandler(
                self.log_dir / f"{name}.log", mode='a'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # JSON structured logging
        self.enable_json = enable_json
        if enable_json:
            self.json_log_file = self.log_dir / f"{name}_structured.jsonl"
        
        # Initialize GPU monitor
        self.gpu_monitor = GPUMonitor()
        
        # Progress tracker (will be set when training starts)
        self.progress_tracker: Optional[ProgressTracker] = None
        
        # Memory alerts
        self.memory_alert_threshold = 0.9  # Alert when GPU memory > 90%
        self.last_memory_alert = 0
        self.memory_alert_cooldown = 60  # seconds
        
        # Log initialization
        self.info("Logger initialized", extra={
            "log_dir": str(self.log_dir),
            "log_level": log_level.value,
            "gpu_info": self.gpu_monitor.get_gpu_info()
        })
    
    def set_progress_tracker(self, total_steps: int, total_epochs: int) -> None:
        """Set up progress tracker for training."""
        self.progress_tracker = ProgressTracker(total_steps, total_epochs)
        self.info("Progress tracker initialized", extra={
            "total_steps": total_steps,
            "total_epochs": total_epochs
        })
    
    def _log_structured(self, level: str, message: str, data: Dict[str, Any]) -> None:
        """Log structured data to JSON file."""
        if not self.enable_json:
            return
        
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "logger": self.name,
            "message": message,
            "data": data
        }
        
        try:
            with open(self.json_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write structured log: {e}")
    
    def _check_memory_alerts(self) -> None:
        """Check for memory usage alerts."""
        memory_info = self.gpu_monitor.get_memory_info()
        utilization = memory_info.get("utilization_percent", 0)
        
        current_time = time.time()
        if (utilization > self.memory_alert_threshold * 100 and 
            current_time - self.last_memory_alert > self.memory_alert_cooldown):
            
            self.warning("High GPU memory usage detected", extra={
                "gpu_memory_utilization": utilization,
                "gpu_memory_allocated": memory_info["allocated"],
                "gpu_memory_free": memory_info["free"]
            })
            self.last_memory_alert = current_time
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        self.logger.debug(message)
        if extra:
            self._log_structured("DEBUG", message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        self.logger.info(message)
        if extra:
            self._log_structured("INFO", message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self.logger.warning(message)
        if extra:
            self._log_structured("WARNING", message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message."""
        self.logger.error(message)
        if extra:
            self._log_structured("ERROR", message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message."""
        self.logger.critical(message)
        if extra:
            self._log_structured("CRITICAL", message, extra)
    
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """Log structured training metrics."""
        # Check for memory alerts
        self._check_memory_alerts()
        
        # Log progress if tracker is available
        progress_info = {}
        if self.progress_tracker:
            progress_info = self.progress_tracker.update_step(metrics.step)
        
        # Create log message
        message = (f"Training Step {metrics.step}: "
                  f"loss={metrics.loss:.4f}, "
                  f"lr={metrics.learning_rate:.2e}, "
                  f"gpu_mem={metrics.gpu_memory_allocated:.2f}GB")
        
        if progress_info:
            message += f", progress={progress_info['progress_percent']:.1f}%, ETA={progress_info['eta_formatted']}"
        
        # Log with structured data
        log_data = {
            **asdict(metrics),
            **progress_info,
            "metric_type": "training"
        }
        
        self.info(message, extra=log_data)
    
    def log_validation_metrics(self, metrics: ValidationMetrics) -> None:
        """Log structured validation metrics."""
        message = (f"Validation Step {metrics.step}: "
                  f"val_loss={metrics.val_loss:.4f}, "
                  f"perplexity={metrics.val_perplexity:.2f}, "
                  f"is_best={metrics.is_best}")
        
        log_data = {
            **asdict(metrics),
            "metric_type": "validation"
        }
        
        self.info(message, extra=log_data)
    
    def log_system_metrics(self) -> SystemMetrics:
        """Log current system metrics."""
        memory_info = self.gpu_monitor.get_memory_info()
        
        metrics = SystemMetrics(
            timestamp=datetime.utcnow().isoformat(),
            gpu_memory_allocated=memory_info["allocated"],
            gpu_memory_reserved=memory_info["reserved"],
            gpu_memory_free=memory_info["free"],
            gpu_memory_total=memory_info["total"],
            gpu_utilization=self.gpu_monitor.get_gpu_utilization(),
            gpu_temperature=self.gpu_monitor.get_gpu_temperature(),
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage_percent=psutil.disk_usage('/').percent
        )
        
        message = (f"System: GPU={metrics.gpu_memory_allocated:.2f}GB/"
                  f"{metrics.gpu_memory_total:.2f}GB, "
                  f"CPU={metrics.cpu_percent:.1f}%, "
                  f"RAM={metrics.memory_percent:.1f}%")
        
        if metrics.gpu_utilization is not None:
            message += f", GPU_util={metrics.gpu_utilization:.1f}%"
        
        if metrics.gpu_temperature is not None:
            message += f", GPU_temp={metrics.gpu_temperature:.1f}°C"
        
        log_data = {
            **asdict(metrics),
            "metric_type": "system"
        }
        
        self.debug(message, extra=log_data)
        return metrics
    
    def log_epoch_summary(self, epoch: int, train_metrics: Dict[str, Any], 
                         val_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Log epoch summary."""
        # Update epoch progress
        progress_info = {}
        if self.progress_tracker:
            progress_info = self.progress_tracker.update_epoch(epoch)
        
        message = f"Epoch {epoch} Summary: train_loss={train_metrics.get('train_loss', 0):.4f}"
        
        if val_metrics:
            message += f", val_loss={val_metrics.get('val_loss', 0):.4f}"
        
        if progress_info:
            message += f", epoch_progress={progress_info['progress_percent']:.1f}%"
        
        log_data = {
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "progress_info": progress_info,
            "metric_type": "epoch_summary"
        }
        
        self.info(message, extra=log_data)
    
    def log_checkpoint_saved(self, checkpoint_path: str, step: int, 
                           loss: float, is_best: bool = False) -> None:
        """Log checkpoint saving."""
        message = f"Checkpoint saved: step={step}, loss={loss:.4f}"
        if is_best:
            message += " (BEST)"
        
        log_data = {
            "checkpoint_path": checkpoint_path,
            "step": step,
            "loss": loss,
            "is_best": is_best,
            "metric_type": "checkpoint"
        }
        
        self.info(message, extra=log_data)
    
    def log_oom_error(self, step: int, old_batch_size: int, new_batch_size: int) -> None:
        """Log out-of-memory error and recovery."""
        message = (f"CUDA OOM at step {step}: "
                  f"reduced batch size {old_batch_size} -> {new_batch_size}")
        
        log_data = {
            "step": step,
            "old_batch_size": old_batch_size,
            "new_batch_size": new_batch_size,
            "gpu_memory": self.gpu_monitor.get_memory_info(),
            "metric_type": "oom_error"
        }
        
        self.warning(message, extra=log_data)
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of logging session."""
        return {
            "logger_name": self.name,
            "log_dir": str(self.log_dir),
            "gpu_info": self.gpu_monitor.get_gpu_info(),
            "current_memory": self.gpu_monitor.get_memory_info(),
            "progress_tracker": {
                "total_steps": self.progress_tracker.total_steps if self.progress_tracker else None,
                "total_epochs": self.progress_tracker.total_epochs if self.progress_tracker else None
            }
        }


def create_training_logger(log_dir: str, 
                          log_level: LogLevel = LogLevel.INFO,
                          enable_console: bool = True) -> StructuredLogger:
    """
    Create a training logger with standard configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        enable_console: Enable console output
        
    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger(
        name="qwen_training",
        log_dir=log_dir,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=True,
        enable_json=True
    )