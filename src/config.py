"""
Configuration management for Qwen pretraining.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import json
import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_huggingface_repo(path: str) -> bool:
    """
    Check if a path is a Hugging Face repository ID.
    
    Args:
        path: Path to check
        
    Returns:
        True if it appears to be a Hugging Face repository ID
    """
    # HF repos typically have format "username/repo-name" and don't exist as local paths
    return (
        "/" in path and 
        not Path(path).exists() and
        not path.startswith("./") and
        not path.startswith("../") and
        not os.path.isabs(path) and
        len(path.split("/")) == 2  # username/repo-name format
    )


@dataclass
class TrainingConfig:
    """Configuration class for Qwen pretraining with validation."""
    
    # Configuration metadata
    title: Optional[str] = None  # Optional title to identify config source
    
    # Model configuration
    model_name: str = "Qwen/Qwen2-0.5B"
    tokenizer_path: str = "./tokenizer"
    data_dir: str = "./dataset"
    
    # Memory optimization
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_sequence_length: int = 512
    
    # Training parameters
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Hardware constraints
    max_gpu_memory_gb: float = 5.5  # Leave buffer for RTX 4050 6GB
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # Monitoring and checkpointing
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "./checkpoints"
    enable_tensorboard: bool = True  # Enable TensorBoard logging
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Additional configuration
    dataloader_num_workers: int = 0  # Set to 0 for Windows compatibility
    save_total_limit: int = 3  # Keep only 3 most recent checkpoints
    verbose: bool = False  # Enable verbose logging
    
    # Testing/debugging options
    max_train_samples: Optional[int] = None  # Limit training samples for testing (None = use all)
    max_val_samples: Optional[int] = None  # Limit validation samples for testing (None = use all)
    
    # Model compatibility options
    allow_vocab_mismatch: bool = True  # Allow vocabulary size mismatch between model and tokenizer
    resize_token_embeddings: bool = True  # Resize model embeddings to match tokenizer
    
    # Early stopping
    early_stopping_enabled: bool = False  # Enable early stopping based on validation loss
    early_stopping_patience: int = 3  # Number of epochs without improvement before stopping
    early_stopping_min_delta: float = 0.0  # Minimum change to qualify as improvement
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate all configuration parameters."""
        self._validate_paths()
        self._validate_training_params()
        self._validate_memory_constraints()
        self._validate_hardware_compatibility()
    
    def _validate_paths(self) -> None:
        """Validate file and directory paths."""
        # Check if tokenizer path is a Hugging Face repository
        if _is_huggingface_repo(self.tokenizer_path):
            # Skip local file validation for Hugging Face repositories
            logger.info(f"Tokenizer path detected as Hugging Face repository: {self.tokenizer_path}")
        else:
            # Validate local tokenizer path
            tokenizer_path = Path(self.tokenizer_path)
            if not tokenizer_path.exists():
                raise ValueError(f"Tokenizer path does not exist: {self.tokenizer_path}")
            
            # Check required tokenizer files
            required_files = ["tokenizer_config.json", "sentencepiece.model"]
            for file_name in required_files:
                file_path = tokenizer_path / file_name
                if not file_path.exists():
                    raise ValueError(f"Required tokenizer file missing: {file_path}")
        
        # Check if data directory is a Hugging Face repository
        if _is_huggingface_repo(self.data_dir):
            # Skip local file validation for Hugging Face repositories
            logger.info(f"Data directory detected as Hugging Face repository: {self.data_dir}")
        else:
            # Validate local data directory
            data_path = Path(self.data_dir)
            if not data_path.exists():
                raise ValueError(f"Data directory does not exist: {self.data_dir}")
            
            # Check for at least train.jsonl
            train_file = data_path / "train.jsonl"
            if not train_file.exists():
                raise ValueError(f"Training data file missing: {train_file}")
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _validate_training_params(self) -> None:
        """Validate training parameters."""
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        if self.gradient_accumulation_steps < 1:
            raise ValueError("Gradient accumulation steps must be at least 1")
        
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        if self.num_epochs < 1:
            raise ValueError("Number of epochs must be at least 1")
        
        if self.max_sequence_length < 1:
            raise ValueError("Max sequence length must be at least 1")
        
        if self.warmup_steps < 0:
            raise ValueError("Warmup steps cannot be negative")
        
        if self.weight_decay < 0:
            raise ValueError("Weight decay cannot be negative")
        
        # Validate early stopping parameters
        if self.early_stopping_patience < 1:
            raise ValueError("Early stopping patience must be at least 1")
        
        if self.early_stopping_min_delta < 0:
            raise ValueError("Early stopping min_delta cannot be negative")
    
    def _validate_memory_constraints(self) -> None:
        """Validate memory-related parameters."""
        if self.max_gpu_memory_gb <= 0:
            raise ValueError("Max GPU memory must be positive")
        
        # Check if batch size and sequence length are reasonable for memory
        estimated_memory_per_sample = (self.max_sequence_length * 4) / (1024**3)  # Rough estimate in GB
        estimated_batch_memory = estimated_memory_per_sample * self.batch_size
        
        if estimated_batch_memory > self.max_gpu_memory_gb * 0.5:  # Use 50% as threshold
            raise ValueError(
                f"Batch size ({self.batch_size}) and sequence length ({self.max_sequence_length}) "
                f"may exceed memory constraints. Estimated memory: {estimated_batch_memory:.2f}GB"
            )
    
    def _validate_hardware_compatibility(self) -> None:
        """Validate hardware compatibility."""
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This training requires GPU support.")
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb < self.max_gpu_memory_gb:
                print(f"Warning: Available GPU memory ({gpu_memory_gb:.1f}GB) is less than "
                      f"configured max memory ({self.max_gpu_memory_gb}GB)")
    
    def get_effective_batch_size(self) -> int:
        """Get the effective batch size considering gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
        # Include title if it exists
        if self.title:
            config_dict['title'] = self.title
        return config_dict
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        # Remove title from config_dict if present (it's metadata only, not a config param)
        config_copy = config_dict.copy()
        title = config_copy.pop('title', None)
        config = cls(**config_copy)
        config.title = title  # Set title separately
        return config
    
    @classmethod
    def from_file(cls, file_path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)