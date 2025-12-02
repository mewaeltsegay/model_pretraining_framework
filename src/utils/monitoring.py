"""
Training monitoring utilities for real-time metrics visualization,
loss curve plotting, and system resource monitoring.
"""
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure
import numpy as np
import psutil
import torch
from collections import deque, defaultdict
from dataclasses import dataclass, asdict

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    step: int
    value: float
    epoch: Optional[int] = None


class MetricsBuffer:
    """Thread-safe buffer for storing metrics with automatic cleanup."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize metrics buffer.
        
        Args:
            max_size: Maximum number of points to keep in memory
        """
        self.max_size = max_size
        self.data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_size))
        self.lock = threading.Lock()
    
    def add_metric(self, name: str, value: float, step: int, epoch: Optional[int] = None) -> None:
        """
        Add a metric point.
        
        Args:
            name: Metric name
            value: Metric value
            step: Training step
            epoch: Optional epoch number
        """
        with self.lock:
            point = MetricPoint(
                timestamp=time.time(),
                step=step,
                value=value,
                epoch=epoch
            )
            self.data[name].append(point)
    
    def get_metric_history(self, name: str, max_points: Optional[int] = None) -> List[MetricPoint]:
        """
        Get metric history.
        
        Args:
            name: Metric name
            max_points: Maximum number of recent points to return
            
        Returns:
            List of metric points
        """
        with self.lock:
            points = list(self.data[name])
            if max_points and len(points) > max_points:
                points = points[-max_points:]
            return points
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        with self.lock:
            if name in self.data and self.data[name]:
                return self.data[name][-1].value
            return None
    
    def get_metric_names(self) -> List[str]:
        """Get all available metric names."""
        with self.lock:
            return list(self.data.keys())
    
    def clear(self) -> None:
        """Clear all metrics."""
        with self.lock:
            self.data.clear()


class SystemResourceMonitor:
    """Monitor system resources (GPU, CPU, memory) in real-time."""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize system resource monitor.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.metrics_buffer = MetricsBuffer()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # GPU handles for nvidia-ml-py
        self.gpu_handles = []
        if PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self.gpu_handles.append(handle)
            except Exception:
                self.gpu_handles = []
    
    def _get_gpu_metrics(self) -> Dict[str, float]:
        """Get GPU metrics."""
        metrics = {}
        
        if torch.cuda.is_available():
            # PyTorch GPU memory
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            
            device_props = torch.cuda.get_device_properties(0)
            total = device_props.total_memory / (1024**3)  # GB
            
            metrics.update({
                "gpu_memory_allocated": allocated,
                "gpu_memory_reserved": reserved,
                "gpu_memory_total": total,
                "gpu_memory_free": total - reserved,
                "gpu_memory_utilization": (allocated / total) * 100 if total > 0 else 0
            })
        
        # NVIDIA-ML metrics
        if self.gpu_handles:
            try:
                handle = self.gpu_handles[0]  # Use first GPU
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics["gpu_utilization"] = utilization.gpu
                metrics["gpu_memory_utilization_nvml"] = utilization.memory
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics["gpu_temperature"] = temp
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    metrics["gpu_power_usage"] = power
                except:
                    pass
                
                # Clock speeds
                try:
                    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    metrics["gpu_graphics_clock"] = graphics_clock
                    metrics["gpu_memory_clock"] = memory_clock
                except:
                    pass
                
            except Exception:
                pass
        
        return metrics
    
    def _get_cpu_metrics(self) -> Dict[str, float]:
        """Get CPU metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
        }
    
    def _get_memory_metrics(self) -> Dict[str, float]:
        """Get system memory metrics."""
        memory = psutil.virtual_memory()
        return {
            "memory_percent": memory.percent,
            "memory_available": memory.available / (1024**3),  # GB
            "memory_used": memory.used / (1024**3),  # GB
            "memory_total": memory.total / (1024**3),  # GB
        }
    
    def _get_disk_metrics(self) -> Dict[str, float]:
        """Get disk usage metrics."""
        disk = psutil.disk_usage('/')
        return {
            "disk_percent": (disk.used / disk.total) * 100,
            "disk_free": disk.free / (1024**3),  # GB
            "disk_used": disk.used / (1024**3),  # GB
            "disk_total": disk.total / (1024**3),  # GB
        }
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        step_counter = 0
        
        while self.monitoring:
            try:
                # Collect all metrics
                all_metrics = {}
                all_metrics.update(self._get_gpu_metrics())
                all_metrics.update(self._get_cpu_metrics())
                all_metrics.update(self._get_memory_metrics())
                all_metrics.update(self._get_disk_metrics())
                
                # Add to buffer
                for name, value in all_metrics.items():
                    self.metrics_buffer.add_metric(name, value, step_counter)
                
                step_counter += 1
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics."""
        metrics = {}
        metrics.update(self._get_gpu_metrics())
        metrics.update(self._get_cpu_metrics())
        metrics.update(self._get_memory_metrics())
        metrics.update(self._get_disk_metrics())
        return metrics
    
    def get_metric_history(self, metric_name: str, duration_minutes: Optional[float] = None) -> List[MetricPoint]:
        """
        Get metric history for a specific duration.
        
        Args:
            metric_name: Name of the metric
            duration_minutes: Duration in minutes (None for all history)
            
        Returns:
            List of metric points
        """
        points = self.metrics_buffer.get_metric_history(metric_name)
        
        if duration_minutes is not None:
            cutoff_time = time.time() - (duration_minutes * 60)
            points = [p for p in points if p.timestamp >= cutoff_time]
        
        return points


class TrainingMetricsVisualizer:
    """Real-time training metrics visualization."""
    
    def __init__(self, save_dir: str, update_interval: float = 5.0):
        """
        Initialize metrics visualizer.
        
        Args:
            save_dir: Directory to save plots
            update_interval: Plot update interval in seconds
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.update_interval = update_interval
        
        self.metrics_buffer = MetricsBuffer()
        self.system_monitor = SystemResourceMonitor()
        
        # Plot configuration
        plt.style.use('default')
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Animation objects
        self.animations = {}
        self.figures = {}
    
    def add_training_metric(self, name: str, value: float, step: int, epoch: Optional[int] = None) -> None:
        """Add a training metric point."""
        self.metrics_buffer.add_metric(name, value, step, epoch)
    
    def create_loss_plot(self, figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """Create loss curve plot."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Training Metrics', fontsize=16)
        
        # Training and validation loss
        ax1 = axes[0, 0]
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2 = axes[0, 1]
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)
        
        # GPU Memory Usage
        ax3 = axes[1, 0]
        ax3.set_title('GPU Memory Usage')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Memory (GB)')
        ax3.grid(True, alpha=0.3)
        
        # Training Speed
        ax4 = axes[1, 1]
        ax4.set_title('Training Speed')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Tokens/Second')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_system_plot(self, figsize: Tuple[int, int] = (12, 8)) -> Figure:
        """Create system resource monitoring plot."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('System Resource Monitoring', fontsize=16)
        
        # GPU Utilization
        ax1 = axes[0, 0]
        ax1.set_title('GPU Utilization')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Utilization (%)')
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        
        # GPU Temperature
        ax2 = axes[0, 1]
        ax2.set_title('GPU Temperature')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Temperature (°C)')
        ax2.grid(True, alpha=0.3)
        
        # GPU Memory
        ax3 = axes[0, 2]
        ax3.set_title('GPU Memory')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Memory (GB)')
        ax3.grid(True, alpha=0.3)
        
        # CPU Usage
        ax4 = axes[1, 0]
        ax4.set_title('CPU Usage')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Usage (%)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        
        # System Memory
        ax5 = axes[1, 1]
        ax5.set_title('System Memory')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Usage (%)')
        ax5.set_ylim(0, 100)
        ax5.grid(True, alpha=0.3)
        
        # Disk Usage
        ax6 = axes[1, 2]
        ax6.set_title('Disk Usage')
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Usage (%)')
        ax6.set_ylim(0, 100)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def update_loss_plot(self, fig: Figure) -> None:
        """Update loss plot with latest data."""
        axes = fig.get_axes()
        
        # Clear axes
        for ax in axes:
            ax.clear()
        
        # Recreate plot structure
        axes[0].set_title('Loss Curves')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title('Learning Rate')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Learning Rate')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_title('GPU Memory Usage')
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Memory (GB)')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].set_title('Training Speed')
        axes[3].set_xlabel('Step')
        axes[3].set_ylabel('Tokens/Second')
        axes[3].grid(True, alpha=0.3)
        
        # Plot training loss
        train_loss_points = self.metrics_buffer.get_metric_history('train_loss')
        if train_loss_points:
            steps = [p.step for p in train_loss_points]
            values = [p.value for p in train_loss_points]
            axes[0].plot(steps, values, label='Train Loss', color=self.colors[0])
        
        # Plot validation loss
        val_loss_points = self.metrics_buffer.get_metric_history('val_loss')
        if val_loss_points:
            steps = [p.step for p in val_loss_points]
            values = [p.value for p in val_loss_points]
            axes[0].plot(steps, values, label='Val Loss', color=self.colors[1])
        
        axes[0].legend()
        
        # Plot learning rate
        lr_points = self.metrics_buffer.get_metric_history('learning_rate')
        if lr_points:
            steps = [p.step for p in lr_points]
            values = [p.value for p in lr_points]
            axes[1].plot(steps, values, color=self.colors[2])
        
        # Plot GPU memory
        gpu_mem_points = self.metrics_buffer.get_metric_history('gpu_memory_allocated')
        if gpu_mem_points:
            steps = [p.step for p in gpu_mem_points]
            values = [p.value for p in gpu_mem_points]
            axes[2].plot(steps, values, color=self.colors[3])
        
        # Plot training speed
        speed_points = self.metrics_buffer.get_metric_history('tokens_per_second')
        if speed_points:
            steps = [p.step for p in speed_points]
            values = [p.value for p in speed_points]
            axes[3].plot(steps, values, color=self.colors[4])
        
        plt.tight_layout()
    
    def update_system_plot(self, fig: Figure) -> None:
        """Update system resource plot with latest data."""
        axes = fig.get_axes()
        
        # Get recent data (last 10 minutes)
        duration_minutes = 10
        
        # Clear and setup axes
        for i, (ax, title, ylabel) in enumerate([
            (axes[0], 'GPU Utilization', 'Utilization (%)'),
            (axes[1], 'GPU Temperature', 'Temperature (°C)'),
            (axes[2], 'GPU Memory', 'Memory (GB)'),
            (axes[3], 'CPU Usage', 'Usage (%)'),
            (axes[4], 'System Memory', 'Usage (%)'),
            (axes[5], 'Disk Usage', 'Usage (%)')
        ]):
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if 'Usage' in ylabel or 'Utilization' in ylabel:
                ax.set_ylim(0, 100)
        
        # Plot metrics
        metric_configs = [
            ('gpu_utilization', axes[0], self.colors[0]),
            ('gpu_temperature', axes[1], self.colors[1]),
            ('gpu_memory_allocated', axes[2], self.colors[2]),
            ('cpu_percent', axes[3], self.colors[3]),
            ('memory_percent', axes[4], self.colors[4]),
            ('disk_percent', axes[5], self.colors[5])
        ]
        
        for metric_name, ax, color in metric_configs:
            points = self.system_monitor.get_metric_history(metric_name, duration_minutes)
            if points:
                times = [datetime.fromtimestamp(p.timestamp) for p in points]
                values = [p.value for p in points]
                ax.plot(times, values, color=color, linewidth=2)
                
                # Format x-axis
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
    
    def save_plots(self, prefix: str = "") -> Dict[str, str]:
        """
        Save current plots to files.
        
        Args:
            prefix: Optional prefix for filenames
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        saved_files = {}
        
        # Create and save loss plot
        loss_fig = self.create_loss_plot()
        self.update_loss_plot(loss_fig)
        loss_path = self.save_dir / f"{prefix}training_metrics.png"
        loss_fig.savefig(loss_path, dpi=150, bbox_inches='tight')
        plt.close(loss_fig)
        saved_files['training_metrics'] = str(loss_path)
        
        # Create and save system plot
        system_fig = self.create_system_plot()
        self.update_system_plot(system_fig)
        system_path = self.save_dir / f"{prefix}system_metrics.png"
        system_fig.savefig(system_path, dpi=150, bbox_inches='tight')
        plt.close(system_fig)
        saved_files['system_metrics'] = str(system_path)
        
        return saved_files
    
    def start_monitoring(self) -> None:
        """Start system resource monitoring."""
        self.system_monitor.start_monitoring()
    
    def stop_monitoring(self) -> None:
        """Stop system resource monitoring."""
        self.system_monitor.stop_monitoring()
    
    def export_metrics_json(self, filepath: str) -> None:
        """
        Export all metrics to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        export_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "training_metrics": {},
            "system_metrics": {}
        }
        
        # Export training metrics
        for metric_name in self.metrics_buffer.get_metric_names():
            points = self.metrics_buffer.get_metric_history(metric_name)
            export_data["training_metrics"][metric_name] = [
                {
                    "timestamp": p.timestamp,
                    "step": p.step,
                    "value": p.value,
                    "epoch": p.epoch
                }
                for p in points
            ]
        
        # Export system metrics
        for metric_name in self.system_monitor.metrics_buffer.get_metric_names():
            points = self.system_monitor.metrics_buffer.get_metric_history(metric_name)
            export_data["system_metrics"][metric_name] = [
                {
                    "timestamp": p.timestamp,
                    "step": p.step,
                    "value": p.value
                }
                for p in points
            ]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


def create_training_monitor(log_dir: str, 
                          update_interval: float = 5.0,
                          start_system_monitoring: bool = True) -> TrainingMetricsVisualizer:
    """
    Create a training monitor with standard configuration.
    
    Args:
        log_dir: Directory for saving plots and logs
        update_interval: Plot update interval in seconds
        start_system_monitoring: Whether to start system monitoring immediately
        
    Returns:
        Configured TrainingMetricsVisualizer instance
    """
    monitor = TrainingMetricsVisualizer(log_dir, update_interval)
    
    if start_system_monitoring:
        monitor.start_monitoring()
    
    return monitor