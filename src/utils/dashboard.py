"""
Real-time training dashboard for monitoring Qwen pretraining progress.
Provides a web-based interface for viewing training metrics and system status.
"""
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver
from urllib.parse import urlparse, parse_qs

from .logger import StructuredLogger
from .monitoring import TrainingMetricsVisualizer, SystemResourceMonitor


class DashboardHandler(SimpleHTTPRequestHandler):
    """HTTP handler for the training dashboard."""
    
    def __init__(self, *args, dashboard_data=None, **kwargs):
        self.dashboard_data = dashboard_data
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_dashboard()
        elif parsed_path.path == '/api/metrics':
            self.serve_metrics()
        elif parsed_path.path == '/api/system':
            self.serve_system_metrics()
        elif parsed_path.path == '/api/status':
            self.serve_status()
        elif parsed_path.path == '/api/logs':
            self.serve_logs()
        else:
            super().do_GET()
    
    def serve_dashboard(self):
        """Serve the main dashboard HTML."""
        html_content = self.get_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_metrics(self):
        """Serve training metrics as JSON."""
        if not self.dashboard_data:
            self.send_error(500, "Dashboard data not available")
            return
        
        metrics = self.dashboard_data.get_training_metrics()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(metrics).encode())
    
    def serve_system_metrics(self):
        """Serve system metrics as JSON."""
        if not self.dashboard_data:
            self.send_error(500, "Dashboard data not available")
            return
        
        metrics = self.dashboard_data.get_system_metrics()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(metrics).encode())
    
    def serve_status(self):
        """Serve training status as JSON."""
        if not self.dashboard_data:
            self.send_error(500, "Dashboard data not available")
            return
        
        status = self.dashboard_data.get_training_status()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())
    
    def serve_logs(self):
        """Serve recent log messages as JSON."""
        if not self.dashboard_data:
            self.send_error(500, "Dashboard data not available")
            return
        
        logs = self.dashboard_data.get_recent_logs()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(logs).encode())
    
    def get_dashboard_html(self):
        """Generate dashboard HTML."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .status-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .status-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .status-label {
            color: #666;
            margin-top: 5px;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            height: 400px;
            position: relative;
        }
        .chart-container canvas {
            max-height: 350px !important;
            width: 100% !important;
        }
        .chart-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
        }
        .alert {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .alert.error {
            background-color: #f8d7da;
            border-color: #f5c6cb;
            color: #721c24;
        }
        .refresh-info {
            text-align: center;
            color: #666;
            margin-top: 20px;
            font-size: 0.9em;
        }
        .log-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-top: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        .log-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .log-entry {
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            margin-bottom: 5px;
            padding: 5px;
            border-radius: 3px;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .log-entry.INFO {
            background-color: #f8f9fa;
            border-left: 3px solid #28a745;
        }
        .log-entry.WARNING {
            background-color: #fff3cd;
            border-left: 3px solid #ffc107;
        }
        .log-entry.ERROR {
            background-color: #f8d7da;
            border-left: 3px solid #dc3545;
        }
        .log-timestamp {
            color: #666;
            font-weight: bold;
        }
        .log-level {
            font-weight: bold;
            margin-right: 5px;
        }
        .log-level.INFO { color: #28a745; }
        .log-level.WARNING { color: #ffc107; }
        .log-level.ERROR { color: #dc3545; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Qwen Training Dashboard</h1>
        <p>Real-time monitoring of model pretraining progress</p>
    </div>

    <div class="status-grid">
        <div class="status-card">
            <div class="status-value" id="current-step">-</div>
            <div class="status-label">Current Step</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="current-epoch">-</div>
            <div class="status-label">Current Epoch</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="current-loss">-</div>
            <div class="status-label">Training Loss</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="gpu-memory">-</div>
            <div class="status-label">GPU Memory (GB)</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="eta">-</div>
            <div class="status-label">ETA</div>
        </div>
        <div class="status-card">
            <div class="status-value" id="gpu-temp">-</div>
            <div class="status-label">GPU Temp (°C)</div>
        </div>
    </div>

    <div class="status-card">
        <div class="status-label">Training Progress</div>
        <div class="progress-bar">
            <div class="progress-fill" id="progress-bar" style="width: 0%"></div>
        </div>
        <div id="progress-text">0% complete</div>
    </div>

    <div id="alerts-container"></div>

    <div class="charts-grid">
        <div class="chart-container">
            <div class="chart-title">Training Loss</div>
            <canvas id="loss-chart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">Learning Rate</div>
            <canvas id="lr-chart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">GPU Memory Usage</div>
            <canvas id="memory-chart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">System Resources</div>
            <canvas id="system-chart"></canvas>
        </div>
    </div>

    <div class="log-container">
        <div class="log-title">📋 Training Logs</div>
        <div id="log-display">
            <div class="log-entry INFO">
                <span class="log-timestamp">--:--:--</span>
                <span class="log-level INFO">INFO</span>
                Waiting for training logs...
            </div>
        </div>
    </div>

    <div class="refresh-info">
        Dashboard updates every 5 seconds | Last update: <span id="last-update">-</span>
    </div>

    <script>
        // Chart configurations
        const chartConfig = {
            type: 'line',
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Training Step'
                        }
                    },
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Value'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                elements: {
                    point: {
                        radius: 2,
                        hoverRadius: 4
                    }
                }
            }
        };

        // Initialize charts
        const lossChart = new Chart(document.getElementById('loss-chart'), {
            ...chartConfig,
            data: {
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: 'rgb(102, 126, 234)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.1
                }]
            }
        });

        const lrChart = new Chart(document.getElementById('lr-chart'), {
            ...chartConfig,
            data: {
                datasets: [{
                    label: 'Learning Rate',
                    data: [],
                    borderColor: 'rgb(255, 127, 14)',
                    backgroundColor: 'rgba(255, 127, 14, 0.1)',
                    tension: 0.1
                }]
            }
        });

        const memoryChart = new Chart(document.getElementById('memory-chart'), {
            ...chartConfig,
            data: {
                datasets: [{
                    label: 'GPU Memory (GB)',
                    data: [],
                    borderColor: 'rgb(44, 160, 44)',
                    backgroundColor: 'rgba(44, 160, 44, 0.1)',
                    tension: 0.1
                }]
            }
        });

        const systemChart = new Chart(document.getElementById('system-chart'), {
            ...chartConfig,
            data: {
                datasets: [
                    {
                        label: 'GPU Utilization (%)',
                        data: [],
                        borderColor: 'rgb(214, 39, 40)',
                        backgroundColor: 'rgba(214, 39, 40, 0.1)',
                        tension: 0.1
                    },
                    {
                        label: 'CPU Usage (%)',
                        data: [],
                        borderColor: 'rgb(148, 103, 189)',
                        backgroundColor: 'rgba(148, 103, 189, 0.1)',
                        tension: 0.1
                    }
                ]
            }
        });

        // Update functions
        function updateStatus(data) {
            document.getElementById('current-step').textContent = data.current_step || '-';
            document.getElementById('current-epoch').textContent = data.current_epoch || '-';
            document.getElementById('current-loss').textContent = data.current_loss ? data.current_loss.toFixed(4) : '-';
            document.getElementById('gpu-memory').textContent = data.gpu_memory ? data.gpu_memory.toFixed(2) : '-';
            document.getElementById('eta').textContent = data.eta || '-';
            document.getElementById('gpu-temp').textContent = data.gpu_temperature ? data.gpu_temperature.toFixed(1) : '-';
            
            if (data.progress_percent !== undefined && data.progress_percent !== null) {
                const progressBar = document.getElementById('progress-bar');
                const progressText = document.getElementById('progress-text');
                const progressValue = Math.max(0, Math.min(100, data.progress_percent));
                progressBar.style.width = progressValue + '%';
                progressText.textContent = progressValue.toFixed(1) + '% complete';
                console.log('Progress updated:', progressValue + '%');
            } else {
                console.log('No progress data received:', data.progress_percent);
            }
        }

        function updateCharts(metricsData, systemData) {
            // Limit data points to prevent infinite scrolling (keep last 100 points)
            const maxPoints = 100;
            
            // Update loss chart
            if (metricsData.train_loss && metricsData.train_loss.length > 0) {
                const lossData = metricsData.train_loss.slice(-maxPoints).map(point => ({
                    x: point.step,
                    y: point.value
                }));
                lossChart.data.datasets[0].data = lossData;
                lossChart.update('none');
            }

            // Update learning rate chart
            if (metricsData.learning_rate && metricsData.learning_rate.length > 0) {
                const lrData = metricsData.learning_rate.slice(-maxPoints).map(point => ({
                    x: point.step,
                    y: point.value
                }));
                lrChart.data.datasets[0].data = lrData;
                lrChart.update('none');
            }

            // Update memory chart
            if (metricsData.gpu_memory_allocated && metricsData.gpu_memory_allocated.length > 0) {
                const memData = metricsData.gpu_memory_allocated.slice(-maxPoints).map(point => ({
                    x: point.step,
                    y: point.value
                }));
                memoryChart.data.datasets[0].data = memData;
                memoryChart.update('none');
            }

            // Update system chart with time-based data
            if (systemData.gpu_utilization && systemData.gpu_utilization.length > 0) {
                const gpuData = systemData.gpu_utilization.slice(-maxPoints).map((point, i) => ({
                    x: point.step || i,
                    y: point.value
                }));
                systemChart.data.datasets[0].data = gpuData;
            }
            
            if (systemData.cpu_percent && systemData.cpu_percent.length > 0) {
                const cpuData = systemData.cpu_percent.slice(-maxPoints).map((point, i) => ({
                    x: point.step || i,
                    y: point.value
                }));
                systemChart.data.datasets[1].data = cpuData;
            }
            
            if (systemData.gpu_utilization || systemData.cpu_percent) {
                systemChart.update('none');
            }
        }

        function showAlert(message, type = 'warning') {
            const container = document.getElementById('alerts-container');
            const alert = document.createElement('div');
            alert.className = `alert ${type}`;
            alert.textContent = message;
            container.appendChild(alert);
            
            // Remove alert after 10 seconds
            setTimeout(() => {
                if (alert.parentNode) {
                    alert.parentNode.removeChild(alert);
                }
            }, 10000);
        }

        function updateLogs(logsData) {
            const logDisplay = document.getElementById('log-display');
            
            if (logsData && logsData.length > 0) {
                // Clear existing logs
                logDisplay.innerHTML = '';
                
                // Add new logs (show last 20 for better performance)
                const recentLogs = logsData.slice(-20);
                recentLogs.forEach(log => {
                    const logEntry = document.createElement('div');
                    logEntry.className = `log-entry ${log.level}`;
                    
                    logEntry.innerHTML = `
                        <span class="log-timestamp">${log.timestamp}</span>
                        <span class="log-level ${log.level}">${log.level}</span>
                        ${log.message}
                    `;
                    
                    logDisplay.appendChild(logEntry);
                });
                
                // Auto-scroll to bottom
                logDisplay.scrollTop = logDisplay.scrollHeight;
            }
        }

        // Fetch and update data
        async function fetchData() {
            try {
                const [statusResponse, metricsResponse, systemResponse, logsResponse] = await Promise.all([
                    fetch('/api/status'),
                    fetch('/api/metrics'),
                    fetch('/api/system'),
                    fetch('/api/logs')
                ]);

                if (statusResponse.ok) {
                    const statusData = await statusResponse.json();
                    console.log('Status data received:', statusData);
                    updateStatus(statusData);
                }

                if (metricsResponse.ok && systemResponse.ok) {
                    const metricsData = await metricsResponse.json();
                    const systemData = await systemResponse.json();
                    updateCharts(metricsData, systemData);
                }

                if (logsResponse.ok) {
                    const logsData = await logsResponse.json();
                    updateLogs(logsData);
                }

                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            } catch (error) {
                console.error('Error fetching data:', error);
                showAlert('Failed to fetch data from server', 'error');
            }
        }

        // Start periodic updates
        fetchData();
        setInterval(fetchData, 5000);
    </script>
</body>
</html>
        """


class DashboardData:
    """Data provider for the training dashboard."""
    
    def __init__(self, logger: StructuredLogger, visualizer: TrainingMetricsVisualizer):
        """
        Initialize dashboard data provider.
        
        Args:
            logger: Structured logger instance
            visualizer: Training metrics visualizer
        """
        self.logger = logger
        self.visualizer = visualizer
        
        # Log buffer for dashboard display
        from collections import deque
        self.log_buffer = deque(maxlen=100)  # Keep last 100 log messages
        
        self.training_status = {
            "is_training": False,
            "current_step": 0,
            "current_epoch": 0,
            "total_steps": 0,
            "total_epochs": 0,
            "start_time": None,
            "current_loss": None,
            "best_loss": None,
            "eta": None,
            "progress_percent": 0
        }
    
    def update_training_status(self, **kwargs) -> None:
        """Update training status."""
        self.training_status.update(kwargs)
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        # Add current system metrics
        system_metrics = self.visualizer.system_monitor.get_current_metrics()
        
        status = self.training_status.copy()
        status.update({
            "gpu_memory": system_metrics.get("gpu_memory_allocated", 0),
            "gpu_temperature": system_metrics.get("gpu_temperature"),
            "gpu_utilization": system_metrics.get("gpu_utilization"),
            "cpu_percent": system_metrics.get("cpu_percent", 0),
            "memory_percent": system_metrics.get("memory_percent", 0)
        })
        
        return status
    
    def get_training_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get training metrics for charts."""
        metrics = {}
        
        # Get metrics from visualizer buffer
        for metric_name in self.visualizer.metrics_buffer.get_metric_names():
            points = self.visualizer.metrics_buffer.get_metric_history(metric_name, max_points=100)
            metrics[metric_name] = [
                {
                    "step": p.step,
                    "value": p.value,
                    "timestamp": p.timestamp,
                    "epoch": p.epoch
                }
                for p in points
            ]
        
        return metrics
    
    def get_system_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get system metrics for charts."""
        metrics = {}
        
        # Get metrics from system monitor
        for metric_name in self.visualizer.system_monitor.metrics_buffer.get_metric_names():
            points = self.visualizer.system_monitor.get_metric_history(metric_name, duration_minutes=10)
            metrics[metric_name] = [
                {
                    "step": p.step,
                    "value": p.value,
                    "timestamp": p.timestamp
                }
                for p in points
            ]
        
        return metrics
    
    def add_log_message(self, level: str, message: str, timestamp: str = None) -> None:
        """Add a log message to the buffer for dashboard display."""
        import datetime
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message
        }
        self.log_buffer.append(log_entry)
    
    def get_recent_logs(self, max_logs: int = 50) -> list:
        """Get recent log messages for dashboard display."""
        return list(self.log_buffer)[-max_logs:]


class TrainingDashboard:
    """Real-time training dashboard server."""
    
    def __init__(self, 
                 logger: StructuredLogger,
                 visualizer: TrainingMetricsVisualizer,
                 port: int = 8080,
                 auto_open: bool = True):
        """
        Initialize training dashboard.
        
        Args:
            logger: Structured logger instance
            visualizer: Training metrics visualizer
            port: Server port
            auto_open: Whether to automatically open browser
        """
        self.logger = logger
        self.visualizer = visualizer
        self.port = port
        self.auto_open = auto_open
        
        self.dashboard_data = DashboardData(logger, visualizer)
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
    
    def start(self) -> None:
        """Start the dashboard server."""
        if self.running:
            return
        
        try:
            # Create server with custom handler
            handler = lambda *args, **kwargs: DashboardHandler(
                *args, dashboard_data=self.dashboard_data, **kwargs
            )
            
            self.server = HTTPServer(('localhost', self.port), handler)
            self.running = True
            
            # Start server in separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            dashboard_url = f"http://localhost:{self.port}"
            self.logger.info(f"Training dashboard started at {dashboard_url}")
            
            # Open browser if requested
            if self.auto_open:
                try:
                    webbrowser.open(dashboard_url)
                except Exception as e:
                    self.logger.warning(f"Could not open browser: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {e}")
            self.running = False
    
    def stop(self) -> None:
        """Stop the dashboard server."""
        if not self.running:
            return
        
        self.running = False
        
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        
        if self.server_thread:
            self.server_thread.join(timeout=2.0)
        
        self.logger.info("Training dashboard stopped")
    
    def update_training_progress(self, 
                               step: int, 
                               epoch: int, 
                               loss: float,
                               learning_rate: float,
                               progress_percent: float,
                               eta: Optional[str] = None) -> None:
        """
        Update training progress information.
        
        Args:
            step: Current training step
            epoch: Current epoch
            loss: Current loss value
            learning_rate: Current learning rate
            progress_percent: Training progress percentage
            eta: Estimated time to completion
        """
        self.dashboard_data.update_training_status(
            current_step=step,
            current_epoch=epoch,
            current_loss=loss,
            progress_percent=progress_percent,
            eta=eta,
            is_training=True
        )
        
        # Add metrics to visualizer
        self.visualizer.add_training_metric("train_loss", loss, step, epoch)
        self.visualizer.add_training_metric("learning_rate", learning_rate, step, epoch)
    
    def update_validation_results(self, val_loss: float, is_best: bool = False) -> None:
        """
        Update validation results.
        
        Args:
            val_loss: Validation loss
            is_best: Whether this is the best validation loss so far
        """
        current_step = self.dashboard_data.training_status["current_step"]
        current_epoch = self.dashboard_data.training_status["current_epoch"]
        
        self.visualizer.add_training_metric("val_loss", val_loss, current_step, current_epoch)
        
        if is_best:
            self.dashboard_data.update_training_status(best_loss=val_loss)
    
    def set_training_config(self, total_steps: int, total_epochs: int) -> None:
        """
        Set training configuration.
        
        Args:
            total_steps: Total number of training steps
            total_epochs: Total number of epochs
        """
        self.dashboard_data.update_training_status(
            total_steps=total_steps,
            total_epochs=total_epochs,
            start_time=datetime.utcnow().isoformat()
        )
    
    def training_completed(self) -> None:
        """Mark training as completed."""
        self.dashboard_data.update_training_status(
            is_training=False,
            progress_percent=100.0,
            eta="Completed"
        )
    
    def add_training_log(self, level: str, message: str) -> None:
        """Add a training log message to the dashboard."""
        self.dashboard_data.add_log_message(level, message)
    
    def get_dashboard_url(self) -> Optional[str]:
        """Get the dashboard URL."""
        if self.running:
            return f"http://localhost:{self.port}"
        return None


def create_training_dashboard(logger: StructuredLogger,
                            visualizer: TrainingMetricsVisualizer,
                            port: int = 8080,
                            auto_open: bool = True,
                            start_immediately: bool = True) -> TrainingDashboard:
    """
    Create and optionally start a training dashboard.
    
    Args:
        logger: Structured logger instance
        visualizer: Training metrics visualizer
        port: Server port
        auto_open: Whether to automatically open browser
        start_immediately: Whether to start the server immediately
        
    Returns:
        TrainingDashboard instance
    """
    dashboard = TrainingDashboard(logger, visualizer, port, auto_open)
    
    if start_immediately:
        dashboard.start()
    
    return dashboard