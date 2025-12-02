"""
Configuration utilities for loading and validating training configurations.
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from ..config import TrainingConfig


def load_config_from_args() -> TrainingConfig:
    """Load configuration from command line arguments."""
    parser = argparse.ArgumentParser(description="Qwen Pretraining Configuration")
    
    # Model configuration
    parser.add_argument("--model-name", type=str, default=None,
                       help="Hugging Face model name")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                       help="Path to tokenizer directory")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Path to dataset directory")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--max-sequence-length", type=int, default=None,
                       help="Maximum sequence length")
    parser.add_argument("--warmup-steps", type=int, default=None,
                       help="Number of warmup steps")
    parser.add_argument("--weight-decay", type=float, default=None,
                       help="Weight decay")
    
    # Memory optimization
    parser.add_argument("--max-gpu-memory-gb", type=float, default=None,
                       help="Maximum GPU memory to use (GB)")
    parser.add_argument("--use-mixed-precision", action="store_true", default=None,
                       help="Use mixed precision training")
    parser.add_argument("--no-mixed-precision", dest="use_mixed_precision", action="store_false",
                       help="Disable mixed precision training")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=None,
                       help="Use gradient checkpointing")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false",
                       help="Disable gradient checkpointing")
    
    # Monitoring and checkpointing
    parser.add_argument("--logging-steps", type=int, default=None,
                       help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=None,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=None,
                       help="Evaluate every N steps")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for checkpoints")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed")
    parser.add_argument("--config", "--config-file", type=str, default=None, dest="config_file",
                       help="Path to configuration file (JSON or YAML)")
    
    args = parser.parse_args()
    
    # If config file is provided, load from file and override with command line args
    if args.config_file:
        import sys
        config = load_config_from_file(args.config_file)
        # Override with command line arguments (only if explicitly provided)
        # Check which arguments were actually provided in sys.argv
        provided_args = set()
        for arg in sys.argv:
            if arg.startswith('--'):
                # Remove leading -- and convert to config key format
                arg_name = arg[2:].replace('-', '_')
                provided_args.add(arg_name)
                # Also check for negative flags
                if arg_name.startswith('no_'):
                    provided_args.add(arg_name[3:])  # Remove 'no_' prefix
        
        for key, value in vars(args).items():
            if key == "config_file":
                continue
            
            # Convert hyphenated keys to underscore
            config_key = key.replace("-", "_")
            
            # Only override if the argument was explicitly provided
            if config_key in provided_args and value is not None:
                if hasattr(config, config_key):
                    setattr(config, config_key, value)
        return config
    else:
        # Create config from command line arguments with defaults
        # Use defaults from TrainingConfig if not provided
        defaults = {
            "model_name": "Qwen/Qwen2-0.5B",
            "tokenizer_path": "./tokenizer",
            "data_dir": "./dataset",
            "batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5e-5,
            "num_epochs": 3,
            "max_sequence_length": 512,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "max_gpu_memory_gb": 5.5,
            "use_mixed_precision": True,
            "gradient_checkpointing": True,
            "logging_steps": 100,
            "save_steps": 1000,
            "eval_steps": 500,
            "output_dir": "./checkpoints",
            "seed": 42,
        }
        
        config_dict = {}
        for key, value in vars(args).items():
            if key != "config_file":
                config_key = key.replace("-", "_")
                # Use provided value or default
                config_dict[config_key] = value if value is not None else defaults.get(config_key)
        
        return TrainingConfig(**config_dict)


def load_config_from_file(file_path: Union[str, Path]) -> TrainingConfig:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        TrainingConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is unsupported
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    # Determine file format from extension
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        # Display title if present
        title = config_dict.get('title')
        if title:
            print(f"\n📋 Loading config: {title}")
            print(f"   From file: {file_path}\n")
    elif file_path.suffix.lower() in ['.yaml', '.yml']:
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    return TrainingConfig.from_dict(config_dict)


def save_config_to_file(config: TrainingConfig, file_path: Union[str, Path]) -> None:
    """
    Save configuration to JSON or YAML file.
    
    Args:
        config: TrainingConfig instance to save
        file_path: Path where to save the configuration
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    if file_path.suffix.lower() == '.json':
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    elif file_path.suffix.lower() in ['.yaml', '.yml']:
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    else:
        # Default to JSON
        with open(file_path.with_suffix('.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)


def validate_environment() -> Dict[str, Any]:
    """
    Validate the environment for training.
    
    Returns:
        Dictionary with environment information and validation results
    """
    import torch
    import platform
    
    env_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "gpu_memory": [],
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            env_info["gpu_names"].append(props.name)
            env_info["gpu_memory"].append(props.total_memory / (1024**3))  # GB
    
    return env_info


def create_default_config_file(file_path: Union[str, Path] = "config.json") -> None:
    """
    Create a default configuration file.
    
    Args:
        file_path: Path where to save the default configuration
    """
    default_config = TrainingConfig()
    save_config_to_file(default_config, file_path)
    print(f"Default configuration saved to: {file_path}")


def merge_configs(base_config: TrainingConfig, override_config: Dict[str, Any]) -> TrainingConfig:
    """
    Merge base configuration with override values.
    
    Args:
        base_config: Base TrainingConfig instance
        override_config: Dictionary with values to override
        
    Returns:
        New TrainingConfig instance with merged values
    """
    base_dict = base_config.to_dict()
    base_dict.update(override_config)
    return TrainingConfig.from_dict(base_dict)


def print_config_summary(config: TrainingConfig) -> None:
    """Print a summary of the configuration."""
    print("=" * 60)
    if config.title:
        print(f"📋 CONFIG TITLE: {config.title}")
        print("   (✅ Config loaded from JSON file)")
        print("=" * 60)
    else:
        print("TRAINING CONFIGURATION SUMMARY")
        print("   (⚠️  Using hardcoded defaults - no JSON config file)")
        print("=" * 60)
    
    print(f"Model: {config.model_name}")
    print(f"Tokenizer: {config.tokenizer_path}")
    print(f"Data Directory: {config.data_dir}")
    print(f"Output Directory: {config.output_dir}")
    print()
    
    print("Training Parameters:")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {config.get_effective_batch_size()}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max Sequence Length: {config.max_sequence_length}")
    print()
    
    print("Memory Optimization:")
    print(f"  Max GPU Memory: {config.max_gpu_memory_gb}GB")
    print(f"  Mixed Precision: {config.use_mixed_precision}")
    print(f"  Gradient Checkpointing: {config.gradient_checkpointing}")
    print()
    
    print("Monitoring:")
    print(f"  Logging Steps: {config.logging_steps}")
    print(f"  Save Steps: {config.save_steps}")
    print(f"  Eval Steps: {config.eval_steps}")
    print("=" * 50)