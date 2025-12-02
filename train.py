#!/usr/bin/env python3
"""
Main training script for Qwen pretraining with comprehensive configuration integration.

This script provides a command-line interface for training the Qwen-0.5B model
with custom datasets and tokenizers, including memory optimization, checkpointing,
and comprehensive error handling.
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import TrainingConfig
from src.utils.config_utils import (
    load_config_from_args, 
    validate_environment, 
    print_config_summary
)
from src.utils.hardware_validator import HardwareValidator, ValidationResult
from src.models.model_manager import ModelManager
from src.data.tokenizer_manager import TokenizerManager
from src.data.data_pipeline import DataPipeline, LanguageModelingCollator
from src.training.training_controller import TrainingController

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Main orchestrator for the training process.
    
    Coordinates all components including model loading, data preparation,
    training execution, and error handling.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the training orchestrator.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.setup_logging()
        
        # Initialize hardware validator
        self.hardware_validator = HardwareValidator()
        
        # Initialize components
        self.model_manager: Optional[ModelManager] = None
        self.tokenizer_manager: Optional[TokenizerManager] = None
        self.data_pipeline: Optional[DataPipeline] = None
        self.training_controller: Optional[TrainingController] = None
        
        # Training state
        self.model = None
        self.datasets = None
        self.dataloaders = None
        
        logger.info("TrainingOrchestrator initialized")
    
    def setup_logging(self) -> None:
        """Set up comprehensive logging configuration."""
        # Create logs directory
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler for all logs
        file_handler = logging.FileHandler(log_dir / "training.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Reduce noise from external libraries
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        
        logger.info(f"Logging configured. Log files in: {log_dir}")
    
    def validate_system_requirements(self) -> ValidationResult:
        """
        Validate system requirements and environment with comprehensive hardware checking.
        
        Returns:
            ValidationResult containing validation results and system information
        """
        logger.info("Running comprehensive system validation...")
        
        try:
            # Run comprehensive pre-training checks
            validation_result = self.hardware_validator.run_pre_training_checks(self.config)
            
            # Print validation report
            self.hardware_validator.print_validation_report(validation_result)
            
            # Log detailed results
            if validation_result.errors:
                for error in validation_result.errors:
                    logger.error(f"Validation error: {error}")
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"Validation warning: {warning}")
            
            if validation_result.suggestions:
                logger.info("Validation suggestions:")
                for suggestion in validation_result.suggestions:
                    logger.info(f"  - {suggestion}")
            
            # Use adjusted configuration if available and validation failed
            if not validation_result.is_valid and validation_result.adjusted_config:
                logger.info("Using auto-adjusted configuration to resolve issues")
                self.config = validation_result.adjusted_config
                
                # Re-run validation with adjusted config
                logger.info("Re-validating with adjusted configuration...")
                validation_result = self.hardware_validator.run_pre_training_checks(self.config)
                
                if validation_result.is_valid:
                    logger.info("✅ Validation passed with adjusted configuration")
                else:
                    logger.error("❌ Validation still failed after auto-adjustment")
            
            # Log system report
            system_report = self.hardware_validator.get_system_report()
            logger.info("System capabilities:")
            for gpu_info in system_report["system_info"]["gpu_info"]:
                logger.info(f"  GPU: {gpu_info['name']} ({gpu_info['total_memory_gb']:.1f}GB)")
            logger.info(f"  RAM: {system_report['system_info']['ram_total_gb']:.1f}GB")
            logger.info(f"  CPU: {system_report['system_info']['cpu_count']} cores")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"System validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                warnings=[],
                errors=[f"System validation failed: {e}"],
                suggestions=["Check system requirements and try again"]
            )
    
    def initialize_components(self) -> None:
        """Initialize all training components."""
        logger.info("Initializing training components...")
        
        try:
            # Initialize tokenizer manager
            logger.info("Initializing tokenizer manager...")
            self.tokenizer_manager = TokenizerManager(tokenizer_path=self.config.tokenizer_path)
            self.tokenizer_manager.load_sentencepiece_tokenizer()
            
            # Initialize model manager
            logger.info("Initializing model manager...")
            self.model_manager = ModelManager(self.config)
            
            # Initialize data pipeline
            logger.info("Initializing data pipeline...")
            self.data_pipeline = DataPipeline(
                data_dir=self.config.data_dir,
                tokenizer_manager=self.tokenizer_manager,
                max_sequence_length=self.config.max_sequence_length,
                max_train_samples=self.config.max_train_samples,
                max_val_samples=self.config.max_val_samples,
                max_test_samples=None  # Test dataset limit not in config yet
            )
            
            # Initialize training controller
            logger.info("Initializing training controller...")
            self.training_controller = TrainingController(self.config)
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize components: {e}") from e
    
    def load_model(self) -> None:
        """Load and configure the model."""
        logger.info("Loading and configuring model...")
        
        try:
            # Load model
            self.model = self.model_manager.load_qwen_model()
            
            # Validate model compatibility with tokenizer
            tokenizer_vocab_size = self.tokenizer_manager.vocab_size
            self.model_manager.validate_model_compatibility(tokenizer_vocab_size)
            
            # Log model information
            model_info = self.model_manager.get_model_info()
            logger.info(f"Model loaded: {model_info['num_parameters']:,} parameters")
            logger.info(f"Model dtype: {model_info['model_dtype']}")
            logger.info(f"Device: {model_info['device']}")
            
            # Check memory constraints
            within_limits, message = self.model_manager.check_memory_constraints()
            if not within_limits:
                logger.warning(f"Memory constraint warning: {message}")
            else:
                logger.info(f"Memory check passed: {message}")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise RuntimeError(f"Failed to load model: {e}") from e
    
    def prepare_datasets(self) -> None:
        """Prepare datasets and data loaders."""
        logger.info("Preparing datasets and data loaders...")
        
        try:
            # Validate datasets
            validation_results = self.data_pipeline.validate_datasets()
            for dataset_name, is_valid in validation_results.items():
                if is_valid:
                    logger.info(f"Dataset '{dataset_name}' validation passed")
                else:
                    logger.warning(f"Dataset '{dataset_name}' validation failed")
            
            # Load datasets
            self.datasets = self.data_pipeline.load_datasets(use_streaming=True)
            
            # Log dataset information
            dataset_info = self.data_pipeline.get_dataset_info()
            for name, info in dataset_info.items():
                if info.get('exists', False):
                    logger.info(f"Dataset '{name}': {info.get('estimated_samples', 'unknown')} samples")
            
            # Create data collator
            collator = LanguageModelingCollator(
                tokenizer_manager=self.tokenizer_manager,
                max_length=self.config.max_sequence_length
            )
            
            # Create data loaders
            self.dataloaders = {}
            
            # Training dataloader
            if 'train' in self.datasets:
                self.dataloaders['train'] = DataLoader(
                    self.datasets['train'],
                    batch_size=self.config.batch_size,
                    collate_fn=collator,
                    num_workers=self.config.dataloader_num_workers,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                logger.info(f"Training dataloader created with batch_size={self.config.batch_size}")
            
            # Validation dataloader
            if 'validation' in self.datasets:
                self.dataloaders['validation'] = DataLoader(
                    self.datasets['validation'],
                    batch_size=self.config.batch_size,
                    collate_fn=collator,
                    num_workers=self.config.dataloader_num_workers,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                logger.info("Validation dataloader created")
            
            # Test dataloader (if needed)
            if 'test' in self.datasets:
                self.dataloaders['test'] = DataLoader(
                    self.datasets['test'],
                    batch_size=self.config.batch_size,
                    collate_fn=collator,
                    num_workers=self.config.dataloader_num_workers,
                    pin_memory=True if torch.cuda.is_available() else False
                )
                logger.info("Test dataloader created")
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            raise RuntimeError(f"Failed to prepare datasets: {e}") from e
    
    def run_training(self) -> Dict[str, Any]:
        """
        Execute the main training loop.
        
        Returns:
            Dictionary containing training results and metrics
        """
        logger.info("Starting training execution...")
        
        try:
            # Get data loaders
            train_dataloader = self.dataloaders['train']
            val_dataloader = self.dataloaders.get('validation', None)
            
            # Run training with integrated checkpointing
            training_results = self.training_controller.train_with_checkpointing(
                model=self.model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                resume_from_checkpoint=True
            )
            
            logger.info("Training completed successfully")
            return training_results
            
        except Exception as e:
            logger.error(f"Training execution failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Training failed: {e}") from e
    
    def handle_training_error(self, error: Exception) -> None:
        """
        Handle training errors with appropriate recovery strategies.
        
        Args:
            error: The exception that occurred
        """
        logger.error(f"Training error occurred: {error}")
        
        # Check if it's a CUDA OOM error
        if "out of memory" in str(error).lower():
            logger.warning("CUDA Out of Memory error detected")
            
            if self.model_manager:
                new_batch_size = self.model_manager.handle_oom_error()
                logger.info(f"Suggested new batch size: {new_batch_size}")
                
                # Update config
                self.config.batch_size = new_batch_size
                logger.info("Consider restarting training with reduced batch size")
        
        # Log memory report if available
        if self.model_manager:
            try:
                memory_report = self.model_manager.get_memory_optimization_report()
                logger.info(f"Memory optimization report: {memory_report}")
            except Exception as e:
                logger.warning(f"Could not generate memory report: {e}")
        
        # Save error information
        error_log_path = Path(self.config.output_dir) / "logs" / "error_report.txt"
        try:
            with open(error_log_path, 'w') as f:
                f.write(f"Training Error Report\n")
                f.write(f"=====================\n\n")
                f.write(f"Error: {error}\n\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n\n")
                f.write(f"Configuration:\n{self.config.to_dict()}\n")
            
            logger.info(f"Error report saved to: {error_log_path}")
        except Exception as e:
            logger.warning(f"Could not save error report: {e}")
    
    def cleanup(self) -> None:
        """Clean up resources and memory."""
        logger.info("Cleaning up resources...")
        
        try:
            if self.model_manager:
                self.model_manager.cleanup()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    def run_complete_training(self) -> Dict[str, Any]:
        """
        Run the complete training pipeline with comprehensive validation and error handling.
        
        Returns:
            Dictionary containing training results
        """
        try:
            # Comprehensive system validation
            validation_result = self.validate_system_requirements()
            if not validation_result.is_valid:
                error_msg = "System validation failed:\n" + "\n".join(validation_result.errors)
                raise RuntimeError(error_msg)
            
            # Log any warnings from validation
            if validation_result.warnings:
                logger.warning("Proceeding with warnings. Monitor training closely.")
            
            # Initialize all components
            self.initialize_components()
            
            # Load model
            self.load_model()
            
            # Prepare datasets
            self.prepare_datasets()
            
            # Final memory check before training
            logger.info("Performing final memory check before training...")
            if self.model_manager:
                memory_report = self.model_manager.get_memory_optimization_report()
                logger.info(f"Pre-training memory status: {memory_report}")
                
                # Auto-optimize if needed
                optimization_summary = self.model_manager.auto_optimize_for_training()
                if optimization_summary.get('batch_size_suggestion'):
                    suggestion = optimization_summary['batch_size_suggestion']
                    logger.info(f"Memory optimizer suggests batch size: {suggestion['suggested']}")
            
            # Run training
            results = self.run_training()
            
            # Log final summary
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Final model saved to: {results['final_model_path']}")
            logger.info(f"Total training steps: {results['total_steps']}")
            logger.info(f"Best validation loss: {results.get('best_val_loss', 'N/A')}")
            
            return results
            
        except Exception as e:
            self.handle_training_error(e)
            raise
        
        finally:
            self.cleanup()


def validate_and_adjust_config(config: TrainingConfig) -> TrainingConfig:
    """
    Validate and potentially adjust configuration before training.
    
    Args:
        config: Original configuration
        
    Returns:
        Validated and potentially adjusted configuration
    """
    print("\n" + "=" * 60)
    print("CONFIGURATION VALIDATION")
    print("=" * 60)
    
    # Create hardware validator for initial check
    validator = HardwareValidator()
    
    # Run validation
    result = validator.run_pre_training_checks(config)
    
    # Print results
    validator.print_validation_report(result)
    
    # Use adjusted config if available and needed
    if result.adjusted_config and (result.errors or result.warnings):
        print("\n🔧 Using auto-adjusted configuration to optimize for your hardware")
        
        # Show key differences
        original_batch_size = config.batch_size
        adjusted_batch_size = result.adjusted_config.batch_size
        
        if original_batch_size != adjusted_batch_size:
            print(f"   Batch size: {original_batch_size} → {adjusted_batch_size}")
        
        original_seq_len = config.max_sequence_length
        adjusted_seq_len = result.adjusted_config.max_sequence_length
        
        if original_seq_len != adjusted_seq_len:
            print(f"   Sequence length: {original_seq_len} → {adjusted_seq_len}")
        
        return result.adjusted_config
    
    return config


def main():
    """Main entry point for the training script."""
    try:
        
        # Load configuration from command line arguments
        config = load_config_from_args()
        
        print("=" * 60)
        print("QWEN PRETRAINING SCRIPT")
        print("=" * 60)
        # Print initial configuration summary
        print_config_summary(config)
        
        # Validate and potentially adjust configuration
        validated_config = validate_and_adjust_config(config)
        
        # Print final configuration if it was adjusted
        if validated_config != config:
            print("\n" + "=" * 60)
            print("FINAL CONFIGURATION (AFTER ADJUSTMENTS)")
            print("=" * 60)
            print_config_summary(validated_config)
        
        # Ask user to confirm if there were significant changes
        if validated_config != config:
            print("\n" + "⚠️  Configuration was automatically adjusted for your hardware.")
            print("Do you want to proceed with the adjusted configuration? (y/n): ", end="")
            
            try:
                response = input().strip().lower()
                if response not in ['y', 'yes']:
                    print("Training cancelled by user.")
                    return 0
            except (EOFError, KeyboardInterrupt):
                print("\nTraining cancelled by user.")
                return 0
        
        # Create training orchestrator with validated config
        orchestrator = TrainingOrchestrator(validated_config)
        
        # Run complete training pipeline
        results = orchestrator.run_complete_training()
        
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Final model: {results['final_model_path']}")
        print("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        print(f"Check logs for detailed error information")
        return 1


if __name__ == "__main__":
    sys.exit(main())