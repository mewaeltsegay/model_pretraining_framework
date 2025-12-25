"""
Training controller for Qwen pretraining with memory-aware batch sizing.
"""
import logging
import random
import time
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
import psutil
import gc

try:
    from ..config import TrainingConfig
    from .checkpoint_manager import CheckpointManager, CheckpointMetadata
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import TrainingConfig
    from training.checkpoint_manager import CheckpointManager, CheckpointMetadata

logger = logging.getLogger(__name__)


class TrainingController:
    """
    Training controller that manages the training loop with memory-aware batch sizing,
    gradient accumulation, and automatic error recovery.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the training controller.
        
        Args:
            config: Training configuration object
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Memory monitoring
        self.initial_memory = None
        self.peak_memory = 0
        
        # Checkpoint management
        self.checkpoint_manager = CheckpointManager(config)
        
        # TensorBoard writer
        self.tensorboard_writer = None
        if config.enable_tensorboard:
            self._setup_tensorboard()
        
        # Set up logging
        self._setup_logging()
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
    
    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducible training."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        
        # Ensure deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"Random seeds set to {self.config.seed} for reproducible training")
    
    def _setup_tensorboard(self) -> None:
        """Set up TensorBoard writer for logging."""
        try:
            log_dir = Path(self.config.output_dir) / "tensorboard_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=str(log_dir))
            logger.info(f"TensorBoard logging enabled. Logs saved to: {log_dir}")
            logger.info(f"  View with: tensorboard --logdir {log_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.tensorboard_writer = None
    
    def setup_training(self, model: AutoModelForCausalLM, train_dataloader: DataLoader) -> Tuple[AdamW, Any]:
        """
        Set up optimizer and learning rate scheduler for training.
        
        Args:
            model: The model to train
            train_dataloader: Training data loader
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        logger.info("Setting up optimizer and scheduler...")
        
        # Calculate total training steps
        # Handle streaming datasets that don't have a length
        try:
            dataloader_len = len(train_dataloader)
            total_steps = dataloader_len * self.config.num_epochs // self.config.gradient_accumulation_steps
            logger.info(f"Dataset length: {dataloader_len}, total steps: {total_steps}")
        except (TypeError, AttributeError) as e:
            # For streaming datasets without known length, estimate based on a reasonable default
            # or use a large number and let the scheduler handle it
            logger.warning(
                f"Cannot determine dataset length ({e}). "
                f"Using estimated total steps based on warmup_steps."
            )
            # Estimate: assume we'll train for at least warmup_steps * 10
            # This is a reasonable default that ensures the scheduler works
            estimated_steps_per_epoch = max(self.config.warmup_steps * 2, 1000)
            total_steps = estimated_steps_per_epoch * self.config.num_epochs // self.config.gradient_accumulation_steps
            logger.info(f"Estimated total steps: {total_steps} (may be adjusted during training)")
        
        # Set up optimizer with weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Set up learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(f"Optimizer: AdamW with lr={self.config.learning_rate}, weight_decay={self.config.weight_decay}")
        logger.info(f"Scheduler: Linear with warmup_steps={self.config.warmup_steps}, total_steps={total_steps}")
        logger.info(f"Effective batch size: {self.config.get_effective_batch_size()}")
        
        return optimizer, scheduler
    
    def _get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "free": 0}
        
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        free = total - reserved
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": total
        }
    
    def _log_memory_usage(self, step_type: str = "training") -> None:
        """Log current memory usage."""
        memory_info = self._get_gpu_memory_info()
        
        logger.info(
            f"[{step_type.upper()}] GPU Memory - "
            f"Allocated: {memory_info['allocated']:.2f}GB, "
            f"Reserved: {memory_info['reserved']:.2f}GB, "
            f"Free: {memory_info['free']:.2f}GB"
        )
        
        # Log to TensorBoard
        if self.tensorboard_writer and step_type in ["training", "epoch_start", "validation_start"]:
            self.tensorboard_writer.add_scalar(f'memory/{step_type}_allocated_gb', memory_info['allocated'], self.global_step)
            self.tensorboard_writer.add_scalar(f'memory/{step_type}_reserved_gb', memory_info['reserved'], self.global_step)
            self.tensorboard_writer.add_scalar(f'memory/{step_type}_free_gb', memory_info['free'], self.global_step)
        
        # Update peak memory tracking
        self.peak_memory = max(self.peak_memory, memory_info['allocated'])
    
    def _clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def train_epoch(self, model: AutoModelForCausalLM, train_dataloader: DataLoader, 
                   optimizer: AdamW, scheduler: Any, epoch: int) -> Dict[str, float]:
        """
        Train the model for one epoch with gradient accumulation and automatic batch size adjustment.
        
        Args:
            model: The model to train
            train_dataloader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        # Initialize memory tracking
        if self.initial_memory is None:
            self.initial_memory = self._get_gpu_memory_info()['allocated']
        
        logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
        self._log_memory_usage("epoch_start")
        
        # Calculate expected batches for this epoch to prevent infinite loops with streaming datasets
        expected_batches = None
        if hasattr(train_dataloader, 'dataset') and hasattr(train_dataloader.dataset, '__len__'):
            dataset_len = len(train_dataloader.dataset)
            if dataset_len > 0:
                expected_batches = (dataset_len + self.config.batch_size - 1) // self.config.batch_size
                logger.info(f"Dataset has {dataset_len} samples, expecting ~{expected_batches} batches per epoch")
        
        # Track batches to prevent infinite loops with streaming datasets
        batches_processed = 0
        
        # Create progress bar for training (updates per batch)
        if expected_batches is not None:
            pbar = tqdm(total=expected_batches, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}", 
                       unit="batch", leave=True, ncols=100)
        else:
            pbar = tqdm(desc=f"Epoch {epoch + 1}/{self.config.num_epochs}", unit="batch", leave=True, ncols=100)
        
        for step, batch in enumerate(train_dataloader):
            # Stop if we've processed all expected batches (for streaming datasets)
            if expected_batches is not None and batches_processed >= expected_batches:
                pbar.close()
                logger.info(f"Processed {batches_processed} batches (expected {expected_batches}). Stopping epoch.")
                break
                
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.config.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                
                # Gradient accumulation step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.use_mixed_precision:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / max(num_batches, 1)
                        current_lr = scheduler.get_last_lr()[0]
                        step_time = time.time() - start_time
                        
                        logger.info(
                            f"Step {self.global_step}: loss={avg_loss:.4f}, "
                            f"lr={current_lr:.2e}, "
                            f"step_time={step_time:.2f}s"
                        )
                        
                        # Log to TensorBoard
                        if self.tensorboard_writer:
                            self.tensorboard_writer.add_scalar('train/loss', avg_loss, self.global_step)
                            self.tensorboard_writer.add_scalar('train/learning_rate', current_lr, self.global_step)
                            self.tensorboard_writer.add_scalar('train/step_time', step_time, self.global_step)
                        
                        self._log_memory_usage("training")
                        start_time = time.time()
                
                num_batches += 1
                batches_processed += 1
                
                # Update progress bar (per batch)
                avg_loss = total_loss / max(num_batches, 1)
                current_lr = scheduler.get_last_lr()[0] if (step + 1) % self.config.gradient_accumulation_steps == 0 else None
                pbar.update(1)
                postfix = {
                    'loss': f'{avg_loss:.4f}',
                    'step': self.global_step
                }
                if current_lr is not None:
                    postfix['lr'] = f'{current_lr:.2e}'
                pbar.set_postfix(postfix)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"CUDA OOM error at step {step}. Attempting recovery...")
                    # Reset gradient scaler state before handling OOM
                    # This prevents "unscale_() has already been called" errors
                    if self.config.use_mixed_precision and self.scaler is not None:
                        try:
                            optimizer.zero_grad()
                            # Reset scaler by calling update() - this resets internal state
                            self.scaler.update()
                        except Exception as scaler_error:
                            logger.warning(f"Could not reset scaler: {scaler_error}. Creating new scaler.")
                            # If update fails, create a new scaler
                            self.scaler = GradScaler()
                    self._handle_oom_error(model, optimizer)
                    continue
                else:
                    raise e
        
        # Close progress bar
        pbar.close()
        
        # Flush any remaining accumulated gradients if we broke early
        if expected_batches is not None and batches_processed >= expected_batches and num_batches % self.config.gradient_accumulation_steps != 0:
            logger.info("Flushing remaining accumulated gradients...")
            if self.config.use_mixed_precision:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            self.global_step += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        metrics = {
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "learning_rate": scheduler.get_last_lr()[0],
            "peak_memory_gb": self.peak_memory
        }
        
        # Log epoch metrics to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('epoch/train_loss', avg_loss, epoch + 1)
            self.tensorboard_writer.add_scalar('epoch/epoch_time', epoch_time, epoch + 1)
            self.tensorboard_writer.add_scalar('epoch/learning_rate', scheduler.get_last_lr()[0], epoch + 1)
            self.tensorboard_writer.add_scalar('epoch/peak_memory_gb', self.peak_memory, epoch + 1)
        
        logger.info(f"Epoch {epoch + 1} completed: {metrics}")
        return metrics
    
    def _handle_oom_error(self, model: AutoModelForCausalLM, optimizer: AdamW) -> None:
        """
        Handle CUDA out of memory errors by reducing batch size and clearing memory.
        
        Args:
            model: The model being trained
            optimizer: The optimizer
        """
        logger.warning("Handling CUDA OOM error...")
        
        # Reset gradient scaler state if using mixed precision
        # This is critical because the scaler might be in an inconsistent state after OOM
        if self.config.use_mixed_precision and self.scaler is not None:
            # The scaler might be in a state where unscale_() was called but update() wasn't
            # We need to reset it by calling update() even if there are no gradients
            # This will reset the scaler's internal state
            try:
                # Clear any pending operations first
                optimizer.zero_grad()
                # Reset scaler by calling update() - this will reset its internal state
                # even if we haven't called step(), it will just reset the scale factor tracking
                self.scaler.update()
            except Exception as e:
                logger.warning(f"Could not reset scaler state: {e}. Creating new scaler.")
                # If update fails, create a new scaler
                self.scaler = GradScaler()
        
        # Clear gradients and memory
        optimizer.zero_grad()
        self._clear_memory()
        
        # Reduce batch size if possible
        if self.config.batch_size > 1:
            old_batch_size = self.config.batch_size
            self.config.batch_size = max(1, self.config.batch_size // 2)
            self.config.gradient_accumulation_steps *= 2  # Maintain effective batch size
            
            logger.warning(
                f"Reduced batch size from {old_batch_size} to {self.config.batch_size}, "
                f"increased gradient accumulation to {self.config.gradient_accumulation_steps}"
            )
        else:
            logger.warning("Cannot reduce batch size further (already at 1)")
        
        self._log_memory_usage("oom_recovery")
    
    def validate_model(self, model: AutoModelForCausalLM, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on validation dataset.
        
        Args:
            model: The model to evaluate
            val_dataloader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        model.eval()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        logger.info("Starting validation...")
        self._log_memory_usage("validation_start")
        
        # Calculate expected validation batches for progress bar
        expected_val_batches = None
        if hasattr(val_dataloader, 'dataset') and hasattr(val_dataloader.dataset, '__len__'):
            dataset_len = len(val_dataloader.dataset)
            if dataset_len > 0:
                expected_val_batches = (dataset_len + self.config.batch_size - 1) // self.config.batch_size
        
        # Create validation progress bar
        if expected_val_batches is not None:
            val_pbar = tqdm(total=expected_val_batches, desc="Validation", unit="batch", leave=False, ncols=100)
        else:
            val_pbar = tqdm(desc="Validation", unit="batch", leave=False, ncols=100)
        
        with torch.no_grad():
            val_batches_processed = 0
            for batch in val_dataloader:
                # Stop if we've processed all expected batches
                if expected_val_batches is not None and val_batches_processed >= expected_val_batches:
                    break
                    
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    
                    # Forward pass with mixed precision
                    with autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
                        outputs = model(**batch)
                        loss = outputs.loss
                    
                    total_loss += loss.item()
                    num_batches += 1
                    val_batches_processed += 1
                    
                    # Update validation progress bar
                    current_val_loss = total_loss / max(num_batches, 1)
                    val_pbar.update(1)
                    val_pbar.set_postfix({'loss': f'{current_val_loss:.4f}'})
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.warning("CUDA OOM during validation. Clearing memory and continuing...")
                        self._clear_memory()
                        continue
                    else:
                        raise e
        
        # Close validation progress bar
        val_pbar.close()
        
        # Calculate validation metrics
        avg_loss = total_loss / max(num_batches, 1)
        val_time = time.time() - start_time
        
        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        metrics = {
            "val_loss": avg_loss,
            "val_perplexity": perplexity,
            "val_time": val_time,
            "num_val_batches": num_batches
        }
        
        # Update best validation loss
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            metrics["is_best"] = True
        else:
            metrics["is_best"] = False
        
        # Log validation metrics to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('val/loss', avg_loss, self.global_step)
            self.tensorboard_writer.add_scalar('val/perplexity', perplexity, self.global_step)
            self.tensorboard_writer.add_scalar('val/val_time', val_time, self.global_step)
            self.tensorboard_writer.add_scalar('val/best_loss', self.best_val_loss, self.global_step)
        
        logger.info(f"Validation completed: {metrics}")
        self._log_memory_usage("validation_end")
        
        return metrics
    
    def log_training_metrics(self, metrics: Dict[str, Any], step: int, metric_type: str = "train") -> None:
        """
        Log training metrics to console and file.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            metric_type: Type of metrics (train, val, etc.)
        """
        # Log to console
        metric_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in metrics.items()])
        logger.info(f"[{metric_type.upper()}] Step {step}: {metric_str}")
        
        # Add system metrics
        memory_info = self._get_gpu_memory_info()
        system_metrics = {
            "gpu_memory_allocated": memory_info["allocated"],
            "gpu_memory_free": memory_info["free"],
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }
        
        # Log system metrics
        system_str = ", ".join([f"{k}={v:.2f}" for k, v in system_metrics.items()])
        logger.info(f"[SYSTEM] {system_str}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of training progress and statistics.
        
        Returns:
            Dictionary containing training summary
        """
        memory_info = self._get_gpu_memory_info()
        
        summary = {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "peak_memory_gb": self.peak_memory,
            "current_memory_gb": memory_info["allocated"],
            "config": self.config.to_dict()
        }
        
        return summary
    
    def save_checkpoint(self, model: AutoModelForCausalLM, optimizer: AdamW, 
                       scheduler: Any, val_loss: Optional[float] = None) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer to save
            scheduler: The scheduler to save
            val_loss: Optional validation loss
            
        Returns:
            Path to saved checkpoint
        """
        return self.checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=self.current_epoch,
            step=self.global_step,
            loss=0.0,  # Will be updated with actual loss
            val_loss=val_loss
        )
    
    def load_checkpoint(self, model: AutoModelForCausalLM, optimizer: AdamW,
                       scheduler: Any, checkpoint_path: Optional[str] = None) -> CheckpointMetadata:
        """
        Load a training checkpoint and resume training state.
        
        Args:
            model: The model to load state into
            optimizer: The optimizer to load state into
            scheduler: The scheduler to load state into
            checkpoint_path: Optional specific checkpoint path
            
        Returns:
            Checkpoint metadata
        """
        metadata = self.checkpoint_manager.load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_path=checkpoint_path
        )
        
        # Update training state from checkpoint
        self.current_epoch = metadata.epoch
        self.global_step = metadata.step
        if metadata.val_loss is not None:
            self.best_val_loss = metadata.val_loss
        
        logger.info(f"Training state restored from checkpoint: epoch={self.current_epoch}, step={self.global_step}")
        return metadata
    
    def save_final_model(self, model: AutoModelForCausalLM, tokenizer=None) -> str:
        """
        Save the final trained model for inference.
        
        Args:
            model: The trained model
            tokenizer: Optional tokenizer to save
            
        Returns:
            Path to saved model
        """
        return self.checkpoint_manager.save_final_model(model, tokenizer)
    
    def should_save_checkpoint(self) -> bool:
        """
        Check if a checkpoint should be saved based on current step.
        
        Returns:
            True if checkpoint should be saved
        """
        return self.global_step % self.config.save_steps == 0
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None
        """
        return self.checkpoint_manager.get_latest_checkpoint()
    
    def train_with_checkpointing(self, model: AutoModelForCausalLM, 
                                train_dataloader: DataLoader, 
                                val_dataloader: Optional[DataLoader] = None,
                                resume_from_checkpoint: bool = True) -> Dict[str, Any]:
        """
        Complete training loop with integrated checkpointing.
        
        Args:
            model: The model to train
            train_dataloader: Training data loader
            val_dataloader: Optional validation data loader
            resume_from_checkpoint: Whether to resume from latest checkpoint
            
        Returns:
            Training summary and metrics
        """
        # Set up optimizer and scheduler
        optimizer, scheduler = self.setup_training(model, train_dataloader)
        
        # Try to resume from checkpoint if requested
        start_epoch = 0
        if resume_from_checkpoint:
            latest_checkpoint = self.get_latest_checkpoint()
            if latest_checkpoint:
                try:
                    metadata = self.load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
                    start_epoch = metadata.epoch
                    logger.info(f"Resumed training from checkpoint at epoch {start_epoch}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
        
        # Training loop
        training_metrics = []
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch_with_checkpointing(
                model, train_dataloader, optimizer, scheduler, epoch
            )
            
            # Validation
            val_metrics = {}
            if val_dataloader is not None:
                # Calculate how often to evaluate based on eval_steps
                # eval_steps is step-based, so convert to epoch-based frequency
                should_eval = False
                try:
                    dataloader_len = len(train_dataloader)
                    if dataloader_len > 0:
                        batches_per_epoch = dataloader_len
                        if self.config.eval_steps >= batches_per_epoch:
                            # Evaluate every N epochs where N = eval_steps / batches_per_epoch
                            epochs_per_eval = max(1, self.config.eval_steps // batches_per_epoch)
                            should_eval = (epoch % epochs_per_eval == 0) or (epoch == self.config.num_epochs - 1)
                        else:
                            # If eval_steps < batches_per_epoch, evaluate every epoch
                            # (since we'll hit eval_steps multiple times per epoch)
                            should_eval = True
                    else:
                        # If we can't determine length, evaluate every epoch
                        should_eval = True
                except (TypeError, AttributeError):
                    # For datasets without length, evaluate every epoch
                    should_eval = True
                
                if should_eval:
                    val_metrics = self.validate_model(model, val_dataloader)
                
                # Save checkpoint if this is the best model
                if val_metrics.get("is_best", False):
                    checkpoint_path = self.save_checkpoint(
                        model, optimizer, scheduler, val_metrics["val_loss"]
                    )
                    logger.info(f"Saved best model checkpoint: {checkpoint_path}")
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics, "epoch": epoch}
            training_metrics.append(epoch_metrics)
            
            # Log epoch summary
            self.log_training_metrics(epoch_metrics, self.global_step, "epoch")
        
        # Save final model
        final_model_path = self.save_final_model(model)
        logger.info(f"Training completed. Final model saved to: {final_model_path}")
        
        # Close TensorBoard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            logger.info("TensorBoard writer closed")
        
        return {
            "training_metrics": training_metrics,
            "final_model_path": final_model_path,
            "total_steps": self.global_step,
            "best_val_loss": self.best_val_loss
        }
    
    def train_epoch_with_checkpointing(self, model: AutoModelForCausalLM, 
                                     train_dataloader: DataLoader,
                                     optimizer: AdamW, scheduler: Any, 
                                     epoch: int) -> Dict[str, float]:
        """
        Train one epoch with integrated checkpoint saving.
        
        Args:
            model: The model to train
            train_dataloader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch number
            
        Returns:
            Training metrics for the epoch
        """
        model.train()
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        # Initialize memory tracking
        if self.initial_memory is None:
            self.initial_memory = self._get_gpu_memory_info()['allocated']
        
        logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
        self._log_memory_usage("epoch_start")
        
        # Calculate expected batches for this epoch to prevent infinite loops with streaming datasets
        expected_batches = None
        if hasattr(train_dataloader, 'dataset') and hasattr(train_dataloader.dataset, '__len__'):
            dataset_len = len(train_dataloader.dataset)
            if dataset_len > 0:
                expected_batches = (dataset_len + self.config.batch_size - 1) // self.config.batch_size
                logger.info(f"Dataset has {dataset_len} samples, expecting ~{expected_batches} batches per epoch")
        
        # Track batches to prevent infinite loops with streaming datasets
        batches_processed = 0
        
        # Create progress bar for training (updates per batch)
        if expected_batches is not None:
            pbar = tqdm(total=expected_batches, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}", 
                       unit="batch", leave=True, ncols=100)
        else:
            pbar = tqdm(desc=f"Epoch {epoch + 1}/{self.config.num_epochs}", unit="batch", leave=True, ncols=100)
        
        for step, batch in enumerate(train_dataloader):
            # Stop if we've processed all expected batches (for streaming datasets)
            if expected_batches is not None and batches_processed >= expected_batches:
                pbar.close()
                logger.info(f"Processed {batches_processed} batches (expected {expected_batches}). Stopping epoch.")
                break
                
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with autocast(device_type='cuda', enabled=self.config.use_mixed_precision):
                    outputs = model(**batch)
                    loss = outputs.loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.config.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                total_loss += loss.item()
                
                # Gradient accumulation step
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.use_mixed_precision:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / max(num_batches, 1)
                        current_lr = scheduler.get_last_lr()[0]
                        step_time = time.time() - start_time
                        
                        logger.info(
                            f"Step {self.global_step}: loss={avg_loss:.4f}, "
                            f"lr={current_lr:.2e}, "
                            f"step_time={step_time:.2f}s"
                        )
                        
                        # Log to TensorBoard
                        if self.tensorboard_writer:
                            self.tensorboard_writer.add_scalar('train/loss', avg_loss, self.global_step)
                            self.tensorboard_writer.add_scalar('train/learning_rate', current_lr, self.global_step)
                            self.tensorboard_writer.add_scalar('train/step_time', step_time, self.global_step)
                        
                        self._log_memory_usage("training")
                        start_time = time.time()
                    
                    # Checkpoint saving
                    if self.should_save_checkpoint():
                        avg_loss = total_loss / max(num_batches, 1)
                        checkpoint_path = self.checkpoint_manager.save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            step=self.global_step,
                            loss=avg_loss,
                            val_loss=None
                        )
                        logger.info(f"Checkpoint saved at step {self.global_step}: {checkpoint_path}")
                
                num_batches += 1
                batches_processed += 1
                
                # Update progress bar (per batch)
                avg_loss = total_loss / max(num_batches, 1)
                current_lr = scheduler.get_last_lr()[0] if (step + 1) % self.config.gradient_accumulation_steps == 0 else None
                pbar.update(1)
                postfix = {
                    'loss': f'{avg_loss:.4f}',
                    'step': self.global_step
                }
                if current_lr is not None:
                    postfix['lr'] = f'{current_lr:.2e}'
                pbar.set_postfix(postfix)
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"CUDA OOM error at step {step}. Attempting recovery...")
                    # Reset gradient scaler state before handling OOM
                    # This prevents "unscale_() has already been called" errors
                    if self.config.use_mixed_precision and self.scaler is not None:
                        try:
                            optimizer.zero_grad()
                            # Reset scaler by calling update() - this resets internal state
                            self.scaler.update()
                        except Exception as scaler_error:
                            logger.warning(f"Could not reset scaler: {scaler_error}. Creating new scaler.")
                            # If update fails, create a new scaler
                            self.scaler = GradScaler()
                    self._handle_oom_error(model, optimizer)
                    continue
                else:
                    raise e
        
        # Close progress bar
        pbar.close()
        
        # Flush any remaining accumulated gradients if we broke early
        if expected_batches is not None and batches_processed >= expected_batches and num_batches % self.config.gradient_accumulation_steps != 0:
            logger.info("Flushing remaining accumulated gradients...")
            if self.config.use_mixed_precision:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            self.global_step += 1
        
        # Calculate epoch metrics
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        metrics = {
            "train_loss": avg_loss,
            "epoch_time": epoch_time,
            "learning_rate": scheduler.get_last_lr()[0],
            "peak_memory_gb": self.peak_memory
        }
        
        # Log epoch metrics to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('epoch/train_loss', avg_loss, epoch + 1)
            self.tensorboard_writer.add_scalar('epoch/epoch_time', epoch_time, epoch + 1)
            self.tensorboard_writer.add_scalar('epoch/learning_rate', scheduler.get_last_lr()[0], epoch + 1)
            self.tensorboard_writer.add_scalar('epoch/peak_memory_gb', self.peak_memory, epoch + 1)
        
        logger.info(f"Epoch {epoch + 1} completed: {metrics}")
        return metrics