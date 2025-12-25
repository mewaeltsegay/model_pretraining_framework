"""
Model management for Qwen pretraining with memory optimization.
"""
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    AutoConfig
)
from typing import Dict, Any, Optional, Tuple
import logging
import gc
import psutil
from pathlib import Path

try:
    from ..config import TrainingConfig
    from .memory_optimizer import MemoryOptimizer, DynamicBatchSizer
except ImportError:
    # Fallback for when running as script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import TrainingConfig
    from models.memory_optimizer import MemoryOptimizer, DynamicBatchSizer

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages Qwen model loading, configuration, and memory optimization.
    
    This class handles:
    - Loading Qwen-0.5B model from Hugging Face
    - Configuring model for causal language modeling pretraining
    - Memory optimization strategies (gradient checkpointing, mixed precision)
    - Model validation and compatibility checks
    - GPU memory monitoring and reporting
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize ModelManager with training configuration.
        
        Args:
            config: TrainingConfig instance with model and training parameters
        """
        self.config = config
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Memory tracking
        self._initial_memory = None
        self._model_memory = None
        
        # Initialize memory optimizer
        self.memory_optimizer = MemoryOptimizer(
            max_gpu_memory_gb=config.max_gpu_memory_gb,
            enable_mixed_precision=config.use_mixed_precision
        )
        
        # Initialize dynamic batch sizer
        self.batch_sizer = DynamicBatchSizer(
            initial_batch_size=config.batch_size,
            min_batch_size=1,
            max_batch_size=min(32, config.batch_size * 4)  # Allow up to 4x initial size
        )
    
    def load_model(self) -> AutoModelForCausalLM:
        """
        Load model using the configured model name.
        
        Returns:
            Loaded AutoModelForCausalLM instance
        """
        return self.load_qwen_model()
        
    def load_qwen_model(self, model_name: Optional[str] = None) -> AutoModelForCausalLM:
        """
        Load Qwen model from Hugging Face Hub.
        
        Args:
            model_name: Optional model name override. Uses config.model_name if None.
            
        Returns:
            Loaded AutoModelForCausalLM instance
            
        Raises:
            RuntimeError: If model loading fails
            ValueError: If model is incompatible with configuration
        """
        model_name = model_name or self.config.model_name
        
        logger.info(f"Loading Qwen model: {model_name}")
        
        # List of alternative model names to try if the primary fails
        alternative_models = [
            model_name,
            "Qwen/Qwen2-0.5B",
            "Qwen/Qwen2-0.5B-Instruct", 
            "microsoft/DialoGPT-small",  # Fallback to a smaller model
            "gpt2"  # Final fallback
        ]
        
        last_error = None
        
        for attempt_model in alternative_models:
            try:
                logger.info(f"Attempting to load model: {attempt_model}")
                
                # Record initial GPU memory (with error handling)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        self._initial_memory = torch.cuda.memory_allocated()
                    except RuntimeError as e:
                        if "busy" in str(e).lower() or "unavailable" in str(e).lower():
                            logger.warning(
                                f"CUDA device busy during model loading initialization: {e}. "
                                f"Continuing without memory tracking."
                            )
                            self._initial_memory = 0
                        else:
                            raise
                
                # Load model configuration first to validate
                model_config = AutoConfig.from_pretrained(attempt_model, trust_remote_code=True)
                logger.info(f"Model config loaded. Vocab size: {model_config.vocab_size}")
                
                # Load model with appropriate settings for pretraining
                # For mixed precision, load in FP32 and use autocast during forward pass
                torch_dtype = torch.float32  # Always load in FP32 for training
                if not self.config.use_mixed_precision:
                    # Only use FP16 loading if not using mixed precision training
                    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                # Try to load with device_map, but fallback to manual device placement if it fails
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        attempt_model,
                        config=model_config,
                        torch_dtype=torch_dtype,
                        device_map="auto" if torch.cuda.is_available() else None,
                        trust_remote_code=True,  # Required for Qwen models
                        low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
                    )
                except Exception as e:
                    logger.warning(f"Failed to load with device_map='auto': {e}. Loading without device_map...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        attempt_model,
                        config=model_config,
                        torch_dtype=torch_dtype,
                        device_map=None,  # Disable device_map
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                
                # Ensure model is on the correct device
                # Check if model is actually on GPU (device_map might have failed)
                model_device = next(self.model.parameters()).device
                if torch.cuda.is_available() and model_device.type != 'cuda':
                    logger.info(f"Model is on {model_device}, moving to {self.device}...")
                    try:
                        self.model = self.model.to(self.device)
                        logger.info(f"Model moved to {self.device}")
                    except RuntimeError as e:
                        if "busy" in str(e).lower() or "unavailable" in str(e).lower():
                            logger.error(
                                f"Cannot move model to GPU: CUDA device is busy or unavailable. "
                                f"Error: {e}. Please check if another process is using the GPU."
                            )
                            raise
                        else:
                            raise
                elif not torch.cuda.is_available():
                    # CPU fallback
                    self.model = self.model.to(self.device)
                
                # Record model memory usage
                if torch.cuda.is_available():
                    self._model_memory = torch.cuda.memory_allocated() - self._initial_memory
                
                logger.info(f"Model loaded successfully: {attempt_model}. Parameters: {self.model.num_parameters():,}")
                
                # Update config with the successfully loaded model name
                self.config.model_name = attempt_model
                
                # Configure for pretraining
                self.configure_for_pretraining()
                
                # Apply memory optimizations (with error handling for CUDA issues)
                try:
                    self.optimize_for_memory()
                except RuntimeError as e:
                    if "busy" in str(e).lower() or "unavailable" in str(e).lower():
                        logger.warning(
                            f"CUDA device is busy during memory optimization: {e}. "
                            f"Continuing without full optimization. "
                            f"This may be due to another process using the GPU."
                        )
                        # Try to at least clear cache without synchronization
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                    else:
                        # Re-raise if it's a different error
                        raise
                
                return self.model
                
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load model {attempt_model}: {str(e)}")
                if attempt_model != alternative_models[-1]:  # Not the last attempt
                    logger.info("Trying next alternative model...")
                    continue
                else:
                    break
        
        # If we get here, all models failed to load
        error_msg = f"Failed to load any model. Last error: {str(last_error)}"
        logger.error(error_msg)
        
        # Provide helpful suggestions
        suggestions = [
            "1. Check your internet connection",
            "2. Try authenticating with Hugging Face: huggingface-cli login",
            "3. Verify the model name exists on Hugging Face Hub",
            "4. Check if you have sufficient disk space for model download",
            "5. Try using a different model name in the configuration"
        ]
        
        logger.error("Troubleshooting suggestions:")
        for suggestion in suggestions:
            logger.error(f"   {suggestion}")
        
        raise RuntimeError(error_msg) from last_error
    
    def configure_for_pretraining(self) -> None:
        """
        Configure the loaded model for causal language modeling pretraining.
        
        This method:
        - Ensures model is in training mode
        - Configures loss function for causal LM
        - Sets up model for gradient computation
        
        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_qwen_model() first.")
        
        logger.info("Configuring model for pretraining")
        
        # Set model to training mode
        self.model.train()
        
        # Ensure gradients are enabled
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Configure model for causal language modeling
        # Qwen models are already configured for causal LM by default
        
        logger.info("Model configured for pretraining")
    
    def optimize_for_memory(self) -> Dict[str, Any]:
        """
        Apply comprehensive memory optimization strategies to the model.
        
        This method applies:
        - Gradient checkpointing if enabled in config
        - Mixed precision setup
        - Automatic memory optimizations based on current usage
        - Memory-efficient attention if available
        
        Returns:
            Dictionary of applied optimizations
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_qwen_model() first.")
        
        logger.info("Applying comprehensive memory optimizations")
        optimizations = {}
        
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                optimizations['gradient_checkpointing'] = True
                logger.info("Gradient checkpointing enabled")
            else:
                logger.warning("Model does not support gradient checkpointing")
                optimizations['gradient_checkpointing'] = False
        
        # Set up mixed precision training
        if self.config.use_mixed_precision:
            scaler = self.memory_optimizer.setup_mixed_precision()
            if scaler is not None:
                optimizations['mixed_precision'] = True
                optimizations['grad_scaler'] = scaler
                
                # For mixed precision, keep model in FP32 and use autocast for forward pass
                # Do not convert model to FP16 as this causes gradient scaling issues
                logger.info("Mixed precision enabled: model stays in FP32, forward pass uses FP16 via autocast")
            else:
                optimizations['mixed_precision'] = False
        
        # Apply automatic memory optimizations
        auto_optimizations = self.memory_optimizer.optimize_memory_automatically(self.model)
        optimizations.update(auto_optimizations)
        
        # Additional memory optimizations
        self._apply_additional_optimizations()
        optimizations['additional_cleanup'] = True
        
        # Monitor memory after optimizations
        memory_stats = self.memory_optimizer.monitor_memory(log_stats=True)
        optimizations['memory_after_optimization'] = memory_stats
        
        logger.info(f"Memory optimizations applied: {optimizations}")
        return optimizations
    
    def _apply_additional_optimizations(self) -> None:
        """Apply additional memory optimization techniques."""
        # Clear any cached computations (with error handling)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "busy" in str(e).lower() or "unavailable" in str(e).lower():
                    logger.debug(f"CUDA cache clear skipped (device busy): {e}")
                else:
                    logger.warning(f"Failed to clear CUDA cache: {e}")
        
        # Force garbage collection (always safe)
        gc.collect()
    
    def validate_model_compatibility(self, tokenizer_vocab_size: int) -> bool:
        """
        Validate model compatibility with tokenizer and configuration.
        
        Args:
            tokenizer_vocab_size: Size of the tokenizer vocabulary
            
        Returns:
            True if model is compatible, False otherwise
            
        Raises:
            RuntimeError: If no model is loaded
            ValueError: If critical compatibility issues are found
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_qwen_model() first.")
        
        logger.info("Validating model compatibility")
        
        # Check vocabulary size compatibility
        model_vocab_size = self.model.config.vocab_size
        if model_vocab_size != tokenizer_vocab_size:
            mismatch_msg = (
                f"Vocabulary size mismatch: model has {model_vocab_size}, "
                f"tokenizer has {tokenizer_vocab_size}"
            )
            
            if not self.config.allow_vocab_mismatch:
                logger.error(mismatch_msg)
                raise ValueError(mismatch_msg)
            
            logger.warning(f"{mismatch_msg}. Proceeding with mismatch handling enabled.")
            
            # Handle vocabulary size mismatch
            if self.config.resize_token_embeddings and tokenizer_vocab_size != model_vocab_size:
                logger.info("Resizing model token embeddings to match tokenizer")
                try:
                    self.model.resize_token_embeddings(tokenizer_vocab_size)
                    logger.info(f"Model embeddings resized from {model_vocab_size} to {tokenizer_vocab_size}")
                    # Update the model config to reflect the new vocab size
                    self.model.config.vocab_size = tokenizer_vocab_size
                except Exception as e:
                    logger.error(f"Failed to resize token embeddings: {e}")
                    logger.warning("Continuing with original model vocabulary size")
            else:
                if tokenizer_vocab_size < model_vocab_size:
                    logger.info(f"Using tokenizer vocab ({tokenizer_vocab_size}) with larger model vocab ({model_vocab_size})")
                    logger.info("Model will only use the first {tokenizer_vocab_size} tokens during training")
                else:
                    logger.info(f"Using model vocab ({model_vocab_size}) with larger tokenizer vocab ({tokenizer_vocab_size})")
                    logger.info("Some tokenizer tokens may not have corresponding embeddings")
        else:
            logger.info(f"Vocabulary sizes match: {model_vocab_size} tokens")
        
        # Check sequence length compatibility
        max_position_embeddings = getattr(self.model.config, 'max_position_embeddings', None)
        if max_position_embeddings and self.config.max_sequence_length > max_position_embeddings:
            logger.warning(
                f"Configured sequence length ({self.config.max_sequence_length}) exceeds "
                f"model's maximum ({max_position_embeddings}). This may cause issues."
            )
        
        # Check if model supports the required features
        if not hasattr(self.model, 'forward'):
            raise ValueError("Model does not have a forward method")
        
        # Validate model is on correct device
        if torch.cuda.is_available():
            model_device = next(self.model.parameters()).device
            if model_device.type != 'cuda':
                logger.warning(f"Model is on {model_device}, but CUDA is available")
        
        logger.info("Model compatibility validation passed")
        return True
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get comprehensive memory usage statistics using the memory optimizer.
        
        Returns:
            Dictionary containing memory usage information in GB
        """
        # Use the memory optimizer for comprehensive stats
        return self.memory_optimizer.get_current_memory_usage()
    
    def check_memory_constraints(self) -> Tuple[bool, str]:
        """
        Check if current memory usage is within configured constraints.
        
        Returns:
            Tuple of (is_within_constraints, status_message)
        """
        memory_stats = self.get_memory_usage()
        
        if not torch.cuda.is_available():
            return True, "No GPU available, memory constraints not applicable"
        
        gpu_used = memory_stats.get('gpu_allocated_gb', 0)
        max_allowed = self.config.max_gpu_memory_gb
        
        if gpu_used > max_allowed:
            return False, f"GPU memory usage ({gpu_used:.2f}GB) exceeds limit ({max_allowed}GB)"
        
        utilization = memory_stats.get('gpu_utilization_percent', 0)
        if utilization > 90:
            return False, f"GPU memory utilization ({utilization:.1f}%) is critically high"
        
        return True, f"Memory usage within constraints ({gpu_used:.2f}GB / {max_allowed}GB)"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded model.
        
        Returns:
            Dictionary containing model information
            
        Raises:
            RuntimeError: If no model is loaded
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_qwen_model() first.")
        
        info = {
            'model_name': self.config.model_name,
            'num_parameters': self.model.num_parameters(),
            'num_trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'model_dtype': str(self.model.dtype),
            'device': str(next(self.model.parameters()).device),
            'vocab_size': self.model.config.vocab_size,
            'hidden_size': getattr(self.model.config, 'hidden_size', 'Unknown'),
            'num_layers': getattr(self.model.config, 'num_hidden_layers', 'Unknown'),
            'gradient_checkpointing': getattr(self.model, 'gradient_checkpointing', False),
        }
        
        # Add memory information
        memory_stats = self.get_memory_usage()
        info.update(memory_stats)
        
        return info
    
    def handle_oom_error(self) -> int:
        """
        Handle CUDA Out of Memory error by adjusting batch size and clearing cache.
        
        Returns:
            New suggested batch size
        """
        logger.warning("Handling CUDA Out of Memory error")
        
        # Use memory optimizer to clear cache
        self.memory_optimizer.clear_memory_cache()
        
        # Adjust batch size using dynamic batch sizer
        new_batch_size = self.batch_sizer.adjust_for_oom()
        
        # Apply additional optimizations if model is loaded
        if self.model is not None:
            self.memory_optimizer.optimize_memory_automatically(self.model)
        
        return new_batch_size
    
    def suggest_optimal_batch_size(self) -> int:
        """
        Suggest optimal batch size based on current memory usage and model.
        
        Returns:
            Suggested batch size
        """
        return self.memory_optimizer.suggest_batch_size(
            current_batch_size=self.batch_sizer.get_current_batch_size(),
            sequence_length=self.config.max_sequence_length
        )
    
    def monitor_memory_usage(self, log_stats: bool = True) -> Dict[str, float]:
        """
        Monitor and log current memory usage.
        
        Args:
            log_stats: Whether to log memory statistics
            
        Returns:
            Current memory statistics
        """
        return self.memory_optimizer.monitor_memory(log_stats=log_stats)
    
    def get_memory_optimization_report(self) -> Dict[str, Any]:
        """
        Get comprehensive memory optimization report.
        
        Returns:
            Dictionary containing memory report and optimization suggestions
        """
        report = self.memory_optimizer.get_memory_report()
        
        # Add model-specific information
        if self.model is not None:
            report['model_info'] = {
                'num_parameters': self.model.num_parameters(),
                'model_dtype': str(self.model.dtype),
                'gradient_checkpointing': getattr(self.model, 'gradient_checkpointing', False),
            }
        
        # Add batch sizing information
        report['batch_sizing'] = {
            'current_batch_size': self.batch_sizer.get_current_batch_size(),
            'initial_batch_size': self.batch_sizer.initial_batch_size,
            'oom_history': len(self.batch_sizer.oom_history),
            'success_history': len(self.batch_sizer.success_history),
        }
        
        return report
    
    def auto_optimize_for_training(self) -> Dict[str, Any]:
        """
        Automatically optimize model and memory settings for training.
        
        This method:
        - Checks current memory constraints
        - Applies optimizations if needed
        - Suggests batch size adjustments
        - Returns summary of actions taken
        
        Returns:
            Dictionary of optimizations and suggestions
        """
        logger.info("Running automatic optimization for training")
        
        optimization_summary = {}
        
        # Check memory constraints
        within_limits, message, suggestions = self.memory_optimizer.check_memory_constraints()
        optimization_summary['memory_check'] = {
            'within_limits': within_limits,
            'message': message,
            'suggestions': suggestions
        }
        
        # Apply model optimizations if needed
        if self.model is not None and (not within_limits or suggestions):
            model_optimizations = self.optimize_for_memory()
            optimization_summary['model_optimizations'] = model_optimizations
        
        # Suggest optimal batch size
        optimal_batch_size = self.suggest_optimal_batch_size()
        current_batch_size = self.batch_sizer.get_current_batch_size()
        
        if optimal_batch_size != current_batch_size:
            optimization_summary['batch_size_suggestion'] = {
                'current': current_batch_size,
                'suggested': optimal_batch_size,
                'action': 'consider_adjustment'
            }
        
        # Log summary
        logger.info(f"Auto-optimization completed: {optimization_summary}")
        
        return optimization_summary
    
    def cleanup(self) -> None:
        """Clean up model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        # Use memory optimizer for thorough cleanup
        self.memory_optimizer.clear_memory_cache()
        
        # Reset batch sizer
        self.batch_sizer.reset()
        
        logger.info("Model cleanup completed")