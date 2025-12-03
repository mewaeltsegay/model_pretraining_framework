"""
Checkpoint management system for Qwen pretraining.
Handles saving, loading, validation, and cleanup of training checkpoints.
"""
import os
import json
import logging
import shutil
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from ..config import TrainingConfig
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for training checkpoints."""
    epoch: int
    step: int
    loss: float
    val_loss: Optional[float]
    learning_rate: float
    timestamp: str
    model_name: str
    config_hash: str
    pytorch_version: str
    cuda_version: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create metadata from dictionary."""
        return cls(**data)


class CheckpointManager:
    """
    Manages training checkpoints including saving, loading, validation, and cleanup.
    
    Handles model state, optimizer state, scheduler state, and training metadata
    with integrity checks and automatic cleanup of old checkpoints.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize checkpoint manager.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.checkpoint_dir = Path(config.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file patterns
        self.checkpoint_pattern = "checkpoint-{step}"
        self.model_file = "pytorch_model.bin"
        self.optimizer_file = "optimizer.pt"
        self.scheduler_file = "scheduler.pt"
        self.metadata_file = "checkpoint_metadata.json"
        self.config_file = "training_config.json"
        
        # Create config hash for validation
        self.config_hash = self._compute_config_hash()
        
        logger.info(f"CheckpointManager initialized with output_dir: {self.checkpoint_dir}")
    
    def _compute_config_hash(self) -> str:
        """Compute hash of training configuration for validation."""
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _get_checkpoint_path(self, step: int) -> Path:
        """Get path for checkpoint directory."""
        checkpoint_name = self.checkpoint_pattern.format(step=step)
        return self.checkpoint_dir / checkpoint_name
    
    def _create_metadata(self, epoch: int, step: int, loss: float, 
                        val_loss: Optional[float], learning_rate: float) -> CheckpointMetadata:
        """Create checkpoint metadata."""
        import datetime
        
        return CheckpointMetadata(
            epoch=epoch,
            step=step,
            loss=loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            timestamp=datetime.datetime.now().isoformat(),
            model_name=self.config.model_name,
            config_hash=self.config_hash,
            pytorch_version=torch.__version__,
            cuda_version=torch.version.cuda if torch.cuda.is_available() else None
        )
    
    def _check_disk_space(self, required_mb: float = 5000.0) -> bool:
        """
        Check if there's enough disk space available.
        
        Args:
            required_mb: Required space in MB (default 5GB)
            
        Returns:
            True if enough space is available
        """
        try:
            stat = shutil.disk_usage(self.checkpoint_dir)
            free_gb = stat.free / (1024 ** 3)
            required_gb = required_mb / 1024
            
            if free_gb < required_gb:
                logger.warning(
                    f"Insufficient disk space: {free_gb:.2f}GB free, "
                    f"{required_gb:.2f}GB required"
                )
                return False
            
            logger.debug(f"Disk space check: {free_gb:.2f}GB free, {required_gb:.2f}GB required")
            return True
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Assume OK if we can't check
    
    def _save_with_retry(self, data: Any, file_path: Path, max_retries: int = 3, 
                        retry_delay: float = 1.0) -> None:
        """
        Save data to file with retry logic.
        
        Args:
            data: Data to save (model state, optimizer state, etc.)
            file_path: Path to save file
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        for attempt in range(max_retries):
            try:
                # Save to temporary file first, then rename (atomic operation)
                temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
                
                # Remove temp file if it exists
                if temp_path.exists():
                    temp_path.unlink()
                
                # Save to temp file
                torch.save(data, temp_path)
                
                # Atomic rename
                temp_path.replace(file_path)
                
                # Verify file was written correctly
                if not file_path.exists() or file_path.stat().st_size == 0:
                    raise RuntimeError(f"File {file_path} is empty or missing after save")
                
                return  # Success
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Failed to save {file_path} (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to save {file_path} after {max_retries} attempts: {e}")
                    raise
    
    def save_checkpoint(self, model: AutoModelForCausalLM, optimizer: Optimizer, 
                       scheduler: _LRScheduler, epoch: int, step: int, loss: float,
                       val_loss: Optional[float] = None) -> str:
        """
        Save a training checkpoint with model, optimizer, and scheduler states.
        
        Args:
            model: The model to save
            optimizer: The optimizer to save
            scheduler: The learning rate scheduler to save
            epoch: Current epoch number
            step: Current training step
            loss: Current training loss
            val_loss: Current validation loss (optional)
            
        Returns:
            Path to the saved checkpoint directory
        """
        checkpoint_path = self._get_checkpoint_path(step)
        
        try:
            # Check disk space before attempting save
            if not self._check_disk_space(required_mb=5000.0):
                raise RuntimeError(
                    "Insufficient disk space to save checkpoint. "
                    "Please free up space or reduce save_total_limit."
                )
            
            # Create checkpoint directory
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving checkpoint at step {step} to {checkpoint_path}")
            
            # Save model state with retry
            model_path = checkpoint_path / self.model_file
            logger.debug(f"Saving model state to {model_path}")
            self._save_with_retry(model.state_dict(), model_path)
            
            # Save optimizer state with retry
            optimizer_path = checkpoint_path / self.optimizer_file
            logger.debug(f"Saving optimizer state to {optimizer_path}")
            self._save_with_retry(optimizer.state_dict(), optimizer_path)
            
            # Save scheduler state with retry
            scheduler_path = checkpoint_path / self.scheduler_file
            logger.debug(f"Saving scheduler state to {scheduler_path}")
            self._save_with_retry(scheduler.state_dict(), scheduler_path)
            
            # Create and save metadata
            metadata = self._create_metadata(epoch, step, loss, val_loss, 
                                           scheduler.get_last_lr()[0])
            metadata_path = checkpoint_path / self.metadata_file
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Save training config
            config_path = checkpoint_path / self.config_file
            self.config.save_to_file(str(config_path))
            
            # Validate checkpoint integrity
            if not self._validate_checkpoint(checkpoint_path):
                raise RuntimeError(f"Checkpoint validation failed for {checkpoint_path}")
            
            logger.info(f"Checkpoint saved successfully: {checkpoint_path}")
            
            # Cleanup old checkpoints (after successful save)
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            # Clean up partial checkpoint on failure
            if checkpoint_path.exists():
                try:
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Cleaned up partial checkpoint: {checkpoint_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up partial checkpoint: {cleanup_error}")
            raise
    
    def load_checkpoint(self, model: AutoModelForCausalLM, optimizer: Optimizer,
                       scheduler: _LRScheduler, checkpoint_path: Optional[str] = None) -> CheckpointMetadata:
        """
        Load a training checkpoint and restore model, optimizer, and scheduler states.
        
        Args:
            model: The model to load state into
            optimizer: The optimizer to load state into
            scheduler: The scheduler to load state into
            checkpoint_path: Path to specific checkpoint (if None, loads latest)
            
        Returns:
            Checkpoint metadata
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError("No checkpoints found")
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Validate checkpoint integrity
        if not self._validate_checkpoint(checkpoint_path):
            raise RuntimeError(f"Checkpoint validation failed for {checkpoint_path}")
        
        try:
            # Load metadata
            metadata_path = checkpoint_path / self.metadata_file
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            metadata = CheckpointMetadata.from_dict(metadata_dict)
            
            # Validate config compatibility
            if metadata.config_hash != self.config_hash:
                logger.warning(
                    f"Config hash mismatch. Current: {self.config_hash}, "
                    f"Checkpoint: {metadata.config_hash}. Proceeding with caution."
                )
            
            # Load model state
            model_path = checkpoint_path / self.model_file
            model_state = torch.load(model_path, map_location='cpu')
            model.load_state_dict(model_state)
            
            # Load optimizer state
            optimizer_path = checkpoint_path / self.optimizer_file
            optimizer_state = torch.load(optimizer_path, map_location='cpu')
            optimizer.load_state_dict(optimizer_state)
            
            # Load scheduler state
            scheduler_path = checkpoint_path / self.scheduler_file
            scheduler_state = torch.load(scheduler_path, map_location='cpu')
            scheduler.load_state_dict(scheduler_state)
            
            logger.info(f"Checkpoint loaded successfully from {checkpoint_path}")
            logger.info(f"Resuming from epoch {metadata.epoch}, step {metadata.step}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def _validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Validate checkpoint integrity and completeness.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            # Check required files exist
            required_files = [
                self.model_file,
                self.optimizer_file,
                self.scheduler_file,
                self.metadata_file,
                self.config_file
            ]
            
            for file_name in required_files:
                file_path = checkpoint_path / file_name
                if not file_path.exists():
                    logger.error(f"Missing checkpoint file: {file_path}")
                    return False
                
                # Check file is not empty
                if file_path.stat().st_size == 0:
                    logger.error(f"Empty checkpoint file: {file_path}")
                    return False
            
            # Validate metadata format
            metadata_path = checkpoint_path / self.metadata_file
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Check required metadata fields
            required_fields = ['epoch', 'step', 'loss', 'learning_rate', 'timestamp']
            for field in required_fields:
                if field not in metadata_dict:
                    logger.error(f"Missing metadata field: {field}")
                    return False
            
            # Try to load PyTorch files to check they're not corrupted
            model_path = checkpoint_path / self.model_file
            torch.load(model_path, map_location='cpu')
            
            optimizer_path = checkpoint_path / self.optimizer_file
            torch.load(optimizer_path, map_location='cpu')
            
            scheduler_path = checkpoint_path / self.scheduler_file
            torch.load(scheduler_path, map_location='cpu')
            
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get the path to the latest checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        
        # Sort by step number (highest first)
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]
    
    def list_checkpoints(self) -> List[Tuple[str, int]]:
        """
        List all available checkpoints.
        
        Returns:
            List of tuples (checkpoint_path, step_number)
        """
        checkpoints = []
        
        if not self.checkpoint_dir.exists():
            return checkpoints
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    # Extract step number from directory name
                    step_str = item.name.split("-")[1]
                    step = int(step_str)
                    
                    # Validate checkpoint
                    if self._validate_checkpoint(item):
                        checkpoints.append((str(item), step))
                    else:
                        logger.warning(f"Invalid checkpoint found: {item}")
                        
                except (ValueError, IndexError):
                    logger.warning(f"Invalid checkpoint directory name: {item.name}")
        
        return checkpoints
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond the save limit."""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self.config.save_total_limit:
            return
        
        # Sort by step number (oldest first)
        checkpoints.sort(key=lambda x: x[1])
        
        # Remove oldest checkpoints
        to_remove = checkpoints[:-self.config.save_total_limit]
        
        for checkpoint_path, step in to_remove:
            try:
                shutil.rmtree(checkpoint_path)
                logger.info(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_path}: {e}")
    
    def save_final_model(self, model: AutoModelForCausalLM, tokenizer: Optional[AutoTokenizer] = None) -> str:
        """
        Save the final trained model in standard format for inference.
        
        Args:
            model: The trained model to save
            tokenizer: Optional tokenizer to save with the model
            
        Returns:
            Path to the saved model directory
        """
        final_model_dir = self.checkpoint_dir / "final_model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Saving final model to {final_model_dir}")
            
            # Save model using Hugging Face format
            model.save_pretrained(final_model_dir)
            
            # Save tokenizer if provided
            if tokenizer is not None:
                tokenizer.save_pretrained(final_model_dir)
            
            # Save training config for reference
            config_path = final_model_dir / "training_config.json"
            self.config.save_to_file(str(config_path))
            
            logger.info(f"Final model saved successfully: {final_model_dir}")
            return str(final_model_dir)
            
        except Exception as e:
            logger.error(f"Failed to save final model: {e}")
            raise
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get detailed information about a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not self._validate_checkpoint(checkpoint_path):
            raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
        
        # Load metadata
        metadata_path = checkpoint_path / self.metadata_file
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get file sizes
        file_sizes = {}
        for file_name in [self.model_file, self.optimizer_file, self.scheduler_file]:
            file_path = checkpoint_path / file_name
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                file_sizes[file_name] = f"{size_mb:.2f} MB"
        
        return {
            "metadata": metadata,
            "file_sizes": file_sizes,
            "checkpoint_path": str(checkpoint_path),
            "is_valid": True
        }
    
    def cleanup_corrupted_checkpoints(self) -> int:
        """
        Remove all corrupted or invalid checkpoints.
        
        Returns:
            Number of corrupted checkpoints removed
        """
        removed_count = 0
        
        if not self.checkpoint_dir.exists():
            return removed_count
        
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                if not self._validate_checkpoint(item):
                    try:
                        shutil.rmtree(item)
                        logger.info(f"Removed corrupted checkpoint: {item}")
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove corrupted checkpoint {item}: {e}")
        
        logger.info(f"Cleanup completed. Removed {removed_count} corrupted checkpoints.")
        return removed_count