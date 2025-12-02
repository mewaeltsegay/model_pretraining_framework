"""
Integration module for enhanced training with comprehensive logging and monitoring.
Provides a wrapper around the training controller with integrated logging, monitoring, and dashboard.
"""
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

try:
    from ..config import TrainingConfig
    from ..training.training_controller import TrainingController
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import TrainingConfig
    from training.training_controller import TrainingController
from .logger import StructuredLogger, TrainingMetrics, ValidationMetrics, create_training_logger, LogLevel
from .monitoring import TrainingMetricsVisualizer, create_training_monitor
from .dashboard import TrainingDashboard, create_training_dashboard


class EnhancedTrainingController:
    """
    Enhanced training controller with integrated logging, monitoring, and dashboard.
    """
    
    def __init__(self, config: TrainingConfig, enable_dashboard: bool = True):
        """
        Initialize enhanced training controller.
        
        Args:
            config: Training configuration
            enable_dashboard: Whether to enable the web dashboard
        """
        self.config = config
        self.enable_dashboard = enable_dashboard
        
        # Create output directories
        self.log_dir = Path(config.output_dir) / "logs"
        self.plots_dir = Path(config.output_dir) / "plots"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging system
        log_level = LogLevel.DEBUG if config.verbose else LogLevel.INFO
        self.logger = create_training_logger(
            log_dir=str(self.log_dir),
            log_level=log_level,
            enable_console=True
        )
        
        # Initialize monitoring and visualization
        self.visualizer = create_training_monitor(
            log_dir=str(self.plots_dir),
            update_interval=5.0,
            start_system_monitoring=True
        )
        
        # Initialize dashboard
        self.dashboard: Optional[TrainingDashboard] = None
        if enable_dashboard:
            self.dashboard = create_training_dashboard(
                logger=self.logger,
                visualizer=self.visualizer,
                port=8080,
                auto_open=True,
                start_immediately=True
            )
            
            # Add custom log handler to capture training logs for dashboard
            self._setup_dashboard_log_handler()
        
        # Initialize base training controller
        self.training_controller = TrainingController(config)
        
        # Training state
        self.training_start_time = None
        self.step_times = []
        self.tokens_processed = 0
        
        self.logger.info("Enhanced training controller initialized", extra={
            "config": config.to_dict(),
            "log_dir": str(self.log_dir),
            "plots_dir": str(self.plots_dir),
            "dashboard_enabled": enable_dashboard,
            "dashboard_url": self.dashboard.get_dashboard_url() if self.dashboard else None
        })
    
    def setup_training(self, model: AutoModelForCausalLM, train_dataloader: DataLoader) -> Tuple[AdamW, Any]:
        """
        Set up training with enhanced logging.
        
        Args:
            model: The model to train
            train_dataloader: Training data loader
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        self.logger.info("Setting up training components...")
        
        # Set up progress tracking
        total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        self.logger.set_progress_tracker(total_steps, self.config.num_epochs)
        
        # Configure dashboard
        if self.dashboard:
            self.dashboard.set_training_config(total_steps, self.config.num_epochs)
        
        # Set up base training controller
        optimizer, scheduler = self.training_controller.setup_training(model, train_dataloader)
        
        # Log model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info("Model setup completed", extra={
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024**2),  # Assuming float32
            "optimizer_type": type(optimizer).__name__,
            "scheduler_type": type(scheduler).__name__,
            "total_training_steps": total_steps
        })
        
        return optimizer, scheduler
    
    def train_epoch_enhanced(self, 
                           model: AutoModelForCausalLM, 
                           train_dataloader: DataLoader,
                           optimizer: AdamW, 
                           scheduler: Any, 
                           epoch: int) -> Dict[str, float]:
        """
        Enhanced training epoch with comprehensive logging and monitoring.
        
        Args:
            model: The model to train
            train_dataloader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            
        Returns:
            Training metrics for the epoch
        """
        if self.training_start_time is None:
            self.training_start_time = time.time()
        
        epoch_start_time = time.time()
        self.logger.info(f"Starting enhanced training epoch {epoch + 1}/{self.config.num_epochs}")
        
        # Log system metrics at epoch start
        system_metrics = self.logger.log_system_metrics()
        
        # Run the actual training epoch with real-time dashboard updates
        epoch_metrics = self._train_epoch_with_dashboard_updates(
            model, train_dataloader, optimizer, scheduler, epoch
        )
        
        # Calculate additional metrics
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - self.training_start_time
        
        # Estimate tokens per second
        if hasattr(train_dataloader.dataset, '__len__'):
            estimated_tokens = len(train_dataloader.dataset) * self.config.max_sequence_length
            tokens_per_second = estimated_tokens / epoch_time if epoch_time > 0 else 0
            self.tokens_processed += estimated_tokens
        else:
            tokens_per_second = 0
        
        # Enhanced metrics
        enhanced_metrics = {
            **epoch_metrics,
            "epoch_time_minutes": epoch_time / 60,
            "total_time_hours": total_time / 3600,
            "tokens_per_second": tokens_per_second,
            "total_tokens_processed": self.tokens_processed,
            "gpu_memory_peak": system_metrics.gpu_memory_allocated,
            "gpu_temperature": system_metrics.gpu_temperature,
            "gpu_utilization": system_metrics.gpu_utilization
        }
        
        # Log epoch summary
        self.logger.log_epoch_summary(epoch, enhanced_metrics)
        
        # Update dashboard
        if self.dashboard:
            # Calculate progress based on steps for more granular updates
            total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
            progress_percent = (self.training_controller.global_step / total_steps) * 100 if total_steps > 0 else 0
            eta = self._calculate_eta(epoch + 1, self.config.num_epochs, epoch_time)
            
            self.dashboard.update_training_progress(
                step=self.training_controller.global_step,
                epoch=epoch,
                loss=enhanced_metrics["train_loss"],
                learning_rate=enhanced_metrics["learning_rate"],
                progress_percent=progress_percent,
                eta=eta
            )
        
        # Save plots periodically
        if (epoch + 1) % max(1, self.config.num_epochs // 10) == 0:
            self._save_training_plots(epoch)
        
        return enhanced_metrics
    
    def _log_training_steps(self, 
                          model: AutoModelForCausalLM, 
                          train_dataloader: DataLoader,
                          optimizer: AdamW, 
                          scheduler: Any, 
                          epoch: int) -> None:
        """Log detailed training step information and update dashboard in real-time."""
        # This method provides step-by-step logging during training
        # We'll override the base training controller to add dashboard updates
        
        # The actual step logging is handled by the base training controller
        # but we can add additional monitoring here by hooking into the logging
        pass
    
    def _train_epoch_with_dashboard_updates(self, 
                                          model: AutoModelForCausalLM, 
                                          train_dataloader: DataLoader,
                                          optimizer: AdamW, 
                                          scheduler: Any, 
                                          epoch: int) -> Dict[str, float]:
        """
        Custom training epoch that updates dashboard in real-time at each logging step.
        """
        import time
        from torch.cuda.amp import autocast
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        # Initialize memory tracking
        if self.training_controller.initial_memory is None:
            self.training_controller.initial_memory = self.training_controller._get_gpu_memory_info()['allocated']
        
        self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
        self.training_controller._log_memory_usage("epoch_start")
        
        for step, batch in enumerate(train_dataloader):
            try:
                # Move batch to device
                batch = {k: v.to(self.training_controller.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with autocast(enabled=self.config.use_mixed_precision):
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.config.use_mixed_precision:
                    self.training_controller.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                
                # Gradient accumulation step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.use_mixed_precision:
                        self.training_controller.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        self.training_controller.scaler.step(optimizer)
                        self.training_controller.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    self.training_controller.global_step += 1
                    
                    # Logging and Dashboard Updates
                    if self.training_controller.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / max(num_batches, 1)
                        current_lr = scheduler.get_last_lr()[0]
                        step_time = time.time() - start_time
                        
                        # Log to console (same as original)
                        self.logger.info(
                            f"Step {self.training_controller.global_step}: loss={avg_loss:.4f}, "
                            f"lr={current_lr:.2e}, "
                            f"step_time={step_time:.2f}s"
                        )
                        
                        # Update dashboard with real-time data
                        if self.dashboard:
                            total_steps = len(train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
                            progress_percent = (self.training_controller.global_step / total_steps) * 100 if total_steps > 0 else 0
                            
                            self.dashboard.update_training_progress(
                                step=self.training_controller.global_step,
                                epoch=epoch,
                                loss=avg_loss,
                                learning_rate=current_lr,
                                progress_percent=progress_percent,
                                eta=self._calculate_eta_from_step_time(step_time, self.training_controller.global_step, total_steps)
                            )
                        
                        self.training_controller._log_memory_usage("training")
                        start_time = time.time()
                
                num_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.logger.warning(f"CUDA OOM error at step {step}. Attempting recovery...")
                    self.training_controller._handle_oom_error(model, optimizer)
                    continue
                else:
                    raise e
        
        # Calculate epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        metrics = {
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "learning_rate": scheduler.get_last_lr()[0],
            "peak_memory_gb": self.training_controller.peak_memory
        }
        
        self.logger.info(f"Epoch {epoch + 1} completed: {metrics}")
        return metrics
    
    def _calculate_eta_from_step_time(self, step_time: float, current_step: int, total_steps: int) -> str:
        """Calculate ETA based on current step time."""
        if current_step == 0:
            return "Calculating..."
        
        remaining_steps = total_steps - current_step
        eta_seconds = remaining_steps * step_time
        eta_delta = timedelta(seconds=int(eta_seconds))
        
        return str(eta_delta)
    
    def validate_model_enhanced(self, 
                              model: AutoModelForCausalLM, 
                              val_dataloader: DataLoader,
                              epoch: int) -> Dict[str, float]:
        """
        Enhanced model validation with comprehensive logging.
        
        Args:
            model: The model to validate
            val_dataloader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Validation metrics
        """
        self.logger.info(f"Starting enhanced validation for epoch {epoch}")
        
        # Run base validation
        val_metrics = self.training_controller.validate_model(model, val_dataloader)
        
        # Create structured validation metrics
        validation_metrics = ValidationMetrics(
            step=self.training_controller.global_step,
            epoch=epoch,
            val_loss=val_metrics["val_loss"],
            val_perplexity=val_metrics["val_perplexity"],
            is_best=val_metrics["is_best"],
            timestamp=datetime.utcnow().isoformat(),
            num_batches=val_metrics["num_val_batches"],
            validation_time=val_metrics["val_time"]
        )
        
        # Log structured validation metrics
        self.logger.log_validation_metrics(validation_metrics)
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.update_validation_results(
                val_loss=val_metrics["val_loss"],
                is_best=val_metrics["is_best"]
            )
        
        # Add to visualizer
        self.visualizer.add_training_metric(
            "val_loss", 
            val_metrics["val_loss"], 
            self.training_controller.global_step, 
            epoch
        )
        self.visualizer.add_training_metric(
            "val_perplexity", 
            val_metrics["val_perplexity"], 
            self.training_controller.global_step, 
            epoch
        )
        
        return val_metrics
    
    def train_with_enhanced_monitoring(self, 
                                     model: AutoModelForCausalLM,
                                     train_dataloader: DataLoader,
                                     val_dataloader: Optional[DataLoader] = None,
                                     resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Complete training loop with enhanced monitoring and logging.
        
        Args:
            model: The model to train
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            resume_from_checkpoint: Whether to resume from checkpoint
            
        Returns:
            Training summary and metrics
        """
        self.logger.info("Starting enhanced training with comprehensive monitoring")
        
        # Set up training
        optimizer, scheduler = self.setup_training(model, train_dataloader)
        
        # Training loop with enhanced monitoring
        training_metrics = []
        
        try:
            for epoch in range(self.config.num_epochs):
                # Train epoch with enhanced logging
                train_metrics = self.train_epoch_enhanced(
                    model, train_dataloader, optimizer, scheduler, epoch
                )
                
                # Validation with enhanced logging
                val_metrics = {}
                if val_dataloader is not None:
                    val_metrics = self.validate_model_enhanced(model, val_dataloader, epoch)
                
                # Combine metrics
                epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
                training_metrics.append(epoch_metrics)
                
                # Save checkpoint if needed
                if self.training_controller.should_save_checkpoint():
                    checkpoint_path = self.training_controller.save_checkpoint(
                        model, optimizer, scheduler, val_metrics.get("val_loss")
                    )
                    self.logger.log_checkpoint_saved(
                        checkpoint_path, 
                        self.training_controller.global_step,
                        train_metrics["train_loss"],
                        val_metrics.get("is_best", False)
                    )
        
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}", extra={"error": str(e)})
            raise
        
        # Training completed
        final_model_path = self.training_controller.save_final_model(model)
        
        # Final plots and exports
        self._save_final_results()
        
        # Update dashboard
        if self.dashboard:
            self.dashboard.training_completed()
            
        # Add final log message
        if self.dashboard:
            self.dashboard.add_training_log("INFO", f"Training completed successfully! Total steps: {self.training_controller.global_step}")
        
        # Training summary
        summary = {
            "training_metrics": training_metrics,
            "final_model_path": final_model_path,
            "total_steps": self.training_controller.global_step,
            "best_val_loss": self.training_controller.best_val_loss,
            "total_training_time": time.time() - self.training_start_time if self.training_start_time else 0,
            "total_tokens_processed": self.tokens_processed,
            "log_dir": str(self.log_dir),
            "plots_dir": str(self.plots_dir),
            "dashboard_url": self.dashboard.get_dashboard_url() if self.dashboard else None
        }
        
        self.logger.info("Enhanced training completed successfully", extra=summary)
        return summary
    
    def _calculate_eta(self, current_epoch: int, total_epochs: int, epoch_time: float) -> str:
        """Calculate estimated time to completion."""
        if current_epoch == 0:
            return "Calculating..."
        
        remaining_epochs = total_epochs - current_epoch
        eta_seconds = remaining_epochs * epoch_time
        eta_delta = timedelta(seconds=int(eta_seconds))
        
        return str(eta_delta)
    
    def _save_training_plots(self, epoch: int) -> None:
        """Save training plots."""
        try:
            plot_files = self.visualizer.save_plots(prefix=f"epoch_{epoch:03d}_")
            self.logger.info(f"Training plots saved for epoch {epoch}", extra={
                "plot_files": plot_files
            })
        except Exception as e:
            self.logger.warning(f"Failed to save training plots: {e}")
    
    def _save_final_results(self) -> None:
        """Save final training results and exports."""
        try:
            # Save final plots
            final_plots = self.visualizer.save_plots(prefix="final_")
            
            # Export metrics to JSON
            metrics_file = self.plots_dir / "training_metrics.json"
            self.visualizer.export_metrics_json(str(metrics_file))
            
            # Get training summary
            summary = self.logger.get_log_summary()
            
            self.logger.info("Final training results saved", extra={
                "final_plots": final_plots,
                "metrics_export": str(metrics_file),
                "log_summary": summary
            })
            
        except Exception as e:
            self.logger.warning(f"Failed to save final results: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            # Stop monitoring
            self.visualizer.stop_monitoring()
            
            # Stop dashboard
            if self.dashboard:
                self.dashboard.stop()
            
            self.logger.info("Enhanced training controller cleanup completed")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
    
    def _setup_dashboard_log_handler(self) -> None:
        """Set up a custom log handler to capture training logs for dashboard display."""
        import logging
        
        class DashboardLogHandler(logging.Handler):
            def __init__(self, dashboard):
                super().__init__()
                self.dashboard = dashboard
                
            def emit(self, record):
                try:
                    # Format the log message
                    message = self.format(record)
                    
                    # Only capture training-related logs
                    if any(keyword in record.name for keyword in ['training', 'models', 'data']):
                        # Extract just the message part (remove timestamp and logger name)
                        if ' - ' in message:
                            parts = message.split(' - ', 2)
                            if len(parts) >= 3:
                                clean_message = parts[2]  # Get the actual message
                            else:
                                clean_message = message
                        else:
                            clean_message = message
                            
                        # Send to dashboard
                        self.dashboard.add_training_log(record.levelname, clean_message)
                except Exception:
                    pass  # Don't let logging errors break training
        
        # Add the handler to the root logger to capture all training logs
        dashboard_handler = DashboardLogHandler(self.dashboard)
        dashboard_handler.setLevel(logging.INFO)
        
        # Add to both the training controller logger and root logger
        logging.getLogger('training').addHandler(dashboard_handler)
        logging.getLogger('models').addHandler(dashboard_handler)
        logging.getLogger('data').addHandler(dashboard_handler)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


def create_enhanced_training_controller(config: TrainingConfig, 
                                      enable_dashboard: bool = True) -> EnhancedTrainingController:
    """
    Create an enhanced training controller with full monitoring capabilities.
    
    Args:
        config: Training configuration
        enable_dashboard: Whether to enable the web dashboard
        
    Returns:
        EnhancedTrainingController instance
    """
    return EnhancedTrainingController(config, enable_dashboard)