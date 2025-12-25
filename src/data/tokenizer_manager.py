"""
TokenizerManager for handling SentencePiece tokenization in the Qwen pretraining pipeline.

This module provides a unified interface for loading and using the custom SentencePiece
tokenizer with proper error handling and validation. Supports both local files and
Hugging Face Hub repositories.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import torch
import sentencepiece as smp

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not available, Hugging Face tokenizer loading disabled")

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available, Hugging Face file download disabled")

logger = logging.getLogger(__name__)


class TokenizerManager:
    """
    Manages SentencePiece tokenizer loading and tokenization operations.
    
    This class handles loading the custom SentencePiece tokenizer from local files,
    provides batch tokenization with proper padding and truncation, and includes
    comprehensive error handling for missing or corrupted tokenizer files.
    """
    
    def __init__(self, tokenizer_path: str = "./tokenizer"):
        """
        Initialize the TokenizerManager.
        
        Args:
            tokenizer_path: Path to the directory containing tokenizer files, 
                          or Hugging Face repository ID (e.g., "username/tokenizer-name")
        """
        self.tokenizer_path_str = tokenizer_path
        self.tokenizer_path = Path(tokenizer_path)
        self.sp_processor = None
        self.hf_tokenizer = None  # For Hugging Face tokenizers
        self.tokenizer_config = None
        self._is_loaded = False
        self._is_hf_tokenizer = False
        
        # Check if this is a Hugging Face repository ID
        # HF repos typically have format "username/repo-name" and don't exist as local paths
        self._is_hf_repo = (
            "/" in tokenizer_path and 
            not self.tokenizer_path.exists() and
            not tokenizer_path.startswith("./") and
            not tokenizer_path.startswith("../") and
            not os.path.isabs(tokenizer_path)
        )
        
        # Expected tokenizer files (only used for local paths)
        if not self._is_hf_repo:
            self.config_file = self.tokenizer_path / "tokenizer_config.json"
            self.model_file = self.tokenizer_path / "sentencepiece.model"
            self.vocab_file = self.tokenizer_path / "sentencepiece.vocab"
        
    def load_sentencepiece_tokenizer(self) -> Union[smp.SentencePieceProcessor, Any]:
        """
        Load the SentencePiece tokenizer from local files or Hugging Face Hub.
        
        Returns:
            SentencePieceProcessor or AutoTokenizer: Loaded tokenizer processor
            
        Raises:
            FileNotFoundError: If required tokenizer files are missing
            ValueError: If tokenizer files are corrupted or invalid
            RuntimeError: If tokenizer loading fails
        """
        # Load from Hugging Face if it's a repository ID
        if self._is_hf_repo:
            return self._load_from_huggingface()
        
        # Otherwise load from local files
        try:
            # Validate tokenizer directory exists
            if not self.tokenizer_path.exists():
                raise FileNotFoundError(
                    f"Tokenizer directory not found: {self.tokenizer_path}"
                )
            
            # Check for required files
            missing_files = []
            for file_path, file_name in [
                (self.config_file, "tokenizer_config.json"),
                (self.model_file, "sentencepiece.model"),
                (self.vocab_file, "sentencepiece.vocab")
            ]:
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                raise FileNotFoundError(
                    f"Missing required tokenizer files: {', '.join(missing_files)}"
                )
            
            # Load tokenizer configuration
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.tokenizer_config = json.load(f)
                logger.info(f"Loaded tokenizer config: vocab_size={self.tokenizer_config.get('vocab_size')}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in tokenizer_config.json: {e}")
            except Exception as e:
                raise RuntimeError(f"Failed to load tokenizer config: {e}")
            
            # Initialize and load SentencePiece processor
            self.sp_processor = smp.SentencePieceProcessor()
            
            try:
                # Load the SentencePiece model
                success = self.sp_processor.load(str(self.model_file))
                if not success:
                    raise RuntimeError("SentencePiece model loading returned False")
                
                # Validate loaded model
                self._validate_tokenizer()
                
                self._is_loaded = True
                logger.info(
                    f"Successfully loaded SentencePiece tokenizer from {self.tokenizer_path}"
                )
                logger.info(
                    f"Tokenizer vocab size: {self.sp_processor.get_piece_size()}"
                )
                
                return self.sp_processor
                
            except Exception as e:
                raise RuntimeError(f"Failed to load SentencePiece model: {e}")
                
        except Exception as e:
            logger.error(f"TokenizerManager initialization failed: {e}")
            raise
    
    def _load_from_huggingface(self):
        """
        Load tokenizer from Hugging Face Hub.
        
        Tries to load as a standalone SentencePiece tokenizer first (by downloading files),
        then falls back to AutoTokenizer if that fails.
        
        Returns:
            SentencePieceProcessor or AutoTokenizer: Loaded tokenizer processor
            
        Raises:
            ImportError: If required libraries are not installed
            RuntimeError: If tokenizer loading fails
        """
        if not HF_HUB_AVAILABLE:
            raise ImportError(
                "huggingface_hub library is required to load tokenizers from Hugging Face. "
                "Install with: pip install huggingface_hub"
            )
        
        try:
            logger.info(f"Loading tokenizer from Hugging Face: {self.tokenizer_path_str}")
            
            # First, try to load as standalone SentencePiece tokenizer
            # Check if sentencepiece.model exists in the repo
            try:
                repo_files = list_repo_files(self.tokenizer_path_str, repo_type="model")
                
                # Check for SentencePiece files
                has_sp_model = "sentencepiece.model" in repo_files
                has_sp_vocab = "sentencepiece.vocab" in repo_files or "sentencepiece.vocab" in [f.lower() for f in repo_files]
                has_config = "tokenizer_config.json" in repo_files
                
                if has_sp_model:
                    logger.info("Detected SentencePiece tokenizer files, downloading...")
                    return self._load_sentencepiece_from_hf(repo_files)
                    
            except Exception as e:
                logger.debug(f"Could not list repo files or load SentencePiece: {e}")
                # Continue to try AutoTokenizer
            
            # Fall back to AutoTokenizer (for model tokenizers)
            if TRANSFORMERS_AVAILABLE:
                logger.info("Attempting to load as model tokenizer with AutoTokenizer...")
                try:
                    self.hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path_str)
                    self._is_hf_tokenizer = True
                    
                    # Load tokenizer config
                    self.tokenizer_config = self.hf_tokenizer.get_vocab() if hasattr(self.hf_tokenizer, 'get_vocab') else {}
                    
                    # Try to get vocab size
                    vocab_size = getattr(self.hf_tokenizer, 'vocab_size', None)
                    if vocab_size:
                        self.tokenizer_config['vocab_size'] = vocab_size
                    
                    logger.info(f"Successfully loaded tokenizer from Hugging Face: {self.tokenizer_path_str}")
                    if vocab_size:
                        logger.info(f"Tokenizer vocab size: {vocab_size}")
                    
                    self._is_loaded = True
                    return self.hf_tokenizer
                except Exception as e2:
                    logger.error(f"AutoTokenizer also failed: {e2}")
                    raise RuntimeError(
                        f"Failed to load tokenizer from Hugging Face. "
                        f"Tried SentencePiece files and AutoTokenizer, both failed. "
                        f"Last error: {e2}"
                    ) from e2
            else:
                raise RuntimeError(
                    "Could not load SentencePiece tokenizer and transformers is not available. "
                    "Install with: pip install transformers"
                )
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer from Hugging Face: {e}")
            raise RuntimeError(f"Failed to load tokenizer from Hugging Face: {e}") from e
    
    def _load_sentencepiece_from_hf(self, repo_files: list):
        """
        Download and load SentencePiece tokenizer files from Hugging Face Hub.
        
        Args:
            repo_files: List of files in the repository
            
        Returns:
            SentencePieceProcessor: Loaded tokenizer processor
        """
        import tempfile
        import shutil
        
        # Create temporary directory for downloaded files
        temp_dir = Path(tempfile.mkdtemp(prefix="hf_tokenizer_"))
        
        try:
            # Download required files
            model_file = None
            vocab_file = None
            config_file = None
            
            # Find sentencepiece.model file (case-insensitive)
            for file in repo_files:
                if file.lower() == "sentencepiece.model":
                    model_file = file
                    break
                elif file.endswith(".model") and "sentencepiece" in file.lower():
                    model_file = file
                    break
            
            # Find sentencepiece.vocab file (case-insensitive)
            for file in repo_files:
                if file.lower() == "sentencepiece.vocab":
                    vocab_file = file
                    break
                elif file.endswith(".vocab") and "sentencepiece" in file.lower():
                    vocab_file = file
                    break
            
            # Find tokenizer_config.json
            if "tokenizer_config.json" in repo_files:
                config_file = "tokenizer_config.json"
            
            if not model_file:
                raise FileNotFoundError("sentencepiece.model file not found in repository")
            
            logger.info(f"Downloading tokenizer files from {self.tokenizer_path_str}...")
            
            # Download files
            local_model_path = hf_hub_download(
                repo_id=self.tokenizer_path_str,
                filename=model_file,
                repo_type="model",
                cache_dir=str(temp_dir)
            )
            
            if vocab_file:
                local_vocab_path = hf_hub_download(
                    repo_id=self.tokenizer_path_str,
                    filename=vocab_file,
                    repo_type="model",
                    cache_dir=str(temp_dir)
                )
            else:
                local_vocab_path = None
                logger.warning("sentencepiece.vocab file not found, continuing without it")
            
            if config_file:
                local_config_path = hf_hub_download(
                    repo_id=self.tokenizer_path_str,
                    filename=config_file,
                    repo_type="model",
                    cache_dir=str(temp_dir)
                )
            else:
                local_config_path = None
                logger.warning("tokenizer_config.json not found, continuing without it")
            
            # Load tokenizer configuration if available
            if local_config_path:
                try:
                    with open(local_config_path, 'r', encoding='utf-8') as f:
                        self.tokenizer_config = json.load(f)
                    logger.info(f"Loaded tokenizer config: vocab_size={self.tokenizer_config.get('vocab_size')}")
                except Exception as e:
                    logger.warning(f"Could not load tokenizer config: {e}")
                    self.tokenizer_config = {}
            else:
                self.tokenizer_config = {}
            
            # Initialize and load SentencePiece processor
            self.sp_processor = smp.SentencePieceProcessor()
            
            # Load the SentencePiece model
            success = self.sp_processor.load(str(local_model_path))
            if not success:
                raise RuntimeError("SentencePiece model loading returned False")
            
            # Validate loaded model
            self._validate_tokenizer()
            
            self._is_loaded = True
            logger.info(
                f"Successfully loaded SentencePiece tokenizer from Hugging Face: {self.tokenizer_path_str}"
            )
            logger.info(
                f"Tokenizer vocab size: {self.sp_processor.get_piece_size()}"
            )
            
            return self.sp_processor
            
        finally:
            # Clean up temporary directory
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Could not clean up temporary directory {temp_dir}: {e}")
    
    def _validate_tokenizer(self) -> None:
        """
        Validate the loaded tokenizer for basic functionality.
        
        Raises:
            ValueError: If tokenizer validation fails
        """
        if not self.sp_processor:
            raise ValueError("Tokenizer not loaded")
        
        try:
            # Test basic tokenization
            test_text = "Hello ሰላም!"
            tokens = self.sp_processor.encode_as_ids(test_text)
            decoded = self.sp_processor.decode_ids(tokens)
            
            if not tokens:
                raise ValueError("Tokenizer produced empty token list")
            
            # Check vocab size consistency
            config_vocab_size = self.tokenizer_config.get('vocab_size')
            actual_vocab_size = self.sp_processor.get_piece_size()
            
            if config_vocab_size and config_vocab_size != actual_vocab_size:
                logger.warning(
                    f"Vocab size mismatch: config={config_vocab_size}, "
                    f"actual={actual_vocab_size}"
                )
            
            # Validate special tokens
            special_tokens = self.tokenizer_config.get('special_tokens', [])
            for token in special_tokens:
                token_id = self.sp_processor.piece_to_id(token)
                if token_id == self.sp_processor.unk_id() and token != '<unk>':
                    logger.warning(f"Special token '{token}' not found in vocabulary")
            
            logger.info("Tokenizer validation passed")
            
        except Exception as e:
            raise ValueError(f"Tokenizer validation failed: {e}")
    
    def tokenize_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = 512,
        padding: Union[bool, str] = True,
        truncation: bool = True,
        return_tensors: str = "pt",
        add_special_tokens: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of texts with proper padding and truncation.
        
        Args:
            texts: List of input texts to tokenize
            max_length: Maximum sequence length (default: 512)
            padding: Whether to pad sequences (default: True)
            truncation: Whether to truncate long sequences (default: True)
            return_tensors: Format of returned tensors ("pt" for PyTorch)
            add_special_tokens: Whether to add BOS/EOS tokens (default: True)
            
        Returns:
            Dictionary containing:
                - input_ids: Token IDs tensor
                - attention_mask: Attention mask tensor
                
        Raises:
            RuntimeError: If tokenizer is not loaded
            ValueError: If input parameters are invalid
        """
        if not self._is_loaded:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        
        # Use Hugging Face tokenizer if available
        if self._is_hf_tokenizer and self.hf_tokenizer:
            return self._tokenize_batch_hf(
                texts, max_length, padding, truncation, return_tensors, add_special_tokens
            )
        
        # Otherwise use SentencePiece processor
        if not self.sp_processor:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        
        if not texts:
            raise ValueError("Empty text list provided")
        
        if max_length and max_length <= 0:
            raise ValueError("max_length must be positive")
        
        try:
            batch_input_ids = []
            
            for text in texts:
                if not isinstance(text, str):
                    raise ValueError(f"All texts must be strings, got {type(text)}")
                
                # Tokenize the text
                token_ids = self.sp_processor.encode_as_ids(text)
                
                # Add special tokens if requested
                if add_special_tokens:
                    bos_id = self.sp_processor.bos_id()
                    eos_id = self.sp_processor.eos_id()
                    
                    if bos_id != -1:  # BOS token exists
                        token_ids = [bos_id] + token_ids
                    if eos_id != -1:  # EOS token exists
                        token_ids = token_ids + [eos_id]
                
                # Apply truncation
                if truncation and max_length and len(token_ids) > max_length:
                    token_ids = token_ids[:max_length]
                    logger.debug(f"Truncated sequence to {max_length} tokens")
                
                batch_input_ids.append(token_ids)
            
            # Apply padding
            if padding:
                pad_id = self.sp_processor.pad_id()
                if pad_id == -1:  # No pad token defined, use unk_id
                    pad_id = self.sp_processor.unk_id()
                    logger.warning("No pad token found, using unk token for padding")
                
                # Find maximum length in batch
                if max_length:
                    target_length = min(max_length, max(len(ids) for ids in batch_input_ids))
                else:
                    target_length = max(len(ids) for ids in batch_input_ids)
                
                # Pad sequences
                padded_input_ids = []
                attention_masks = []
                
                for token_ids in batch_input_ids:
                    # Create attention mask (1 for real tokens, 0 for padding)
                    attention_mask = [1] * len(token_ids)
                    
                    # Pad if necessary
                    if len(token_ids) < target_length:
                        padding_length = target_length - len(token_ids)
                        token_ids = token_ids + [pad_id] * padding_length
                        attention_mask = attention_mask + [0] * padding_length
                    
                    padded_input_ids.append(token_ids)
                    attention_masks.append(attention_mask)
                
                batch_input_ids = padded_input_ids
            else:
                # No padding - create attention masks with all 1s
                attention_masks = [[1] * len(ids) for ids in batch_input_ids]
            
            # Convert to tensors
            result = {}
            if return_tensors == "pt":
                result["input_ids"] = torch.tensor(batch_input_ids, dtype=torch.long)
                result["attention_mask"] = torch.tensor(attention_masks, dtype=torch.long)
            else:
                result["input_ids"] = batch_input_ids
                result["attention_mask"] = attention_masks
            
            logger.debug(
                f"Tokenized batch of {len(texts)} texts, "
                f"shape: {result['input_ids'].shape if return_tensors == 'pt' else 'list'}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Batch tokenization failed: {e}")
            raise RuntimeError(f"Tokenization failed: {e}")
    
    def _tokenize_batch_hf(
        self,
        texts: List[str],
        max_length: Optional[int],
        padding: Union[bool, str],
        truncation: bool,
        return_tensors: str,
        add_special_tokens: bool
    ) -> Dict[str, torch.Tensor]:
        """Tokenize batch using Hugging Face tokenizer."""
        try:
            encoded = self.hf_tokenizer(
                texts,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
                add_special_tokens=add_special_tokens
            )
            return encoded
        except Exception as e:
            logger.error(f"Hugging Face tokenization failed: {e}")
            raise RuntimeError(f"Tokenization failed: {e}") from e
    
    def get_tokenizer_config(self) -> Dict[str, Any]:
        """
        Get the tokenizer configuration.
        
        Returns:
            Dictionary containing tokenizer configuration
            
        Raises:
            RuntimeError: If tokenizer is not loaded
        """
        if not self._is_loaded:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        
        config = self.tokenizer_config.copy()
        
        # Add runtime information
        if self.sp_processor:
            config.update({
                "actual_vocab_size": self.sp_processor.get_piece_size(),
                "bos_id": self.sp_processor.bos_id(),
                "eos_id": self.sp_processor.eos_id(),
                "unk_id": self.sp_processor.unk_id(),
                "pad_id": self.sp_processor.pad_id(),
            })
        
        return config
    
    def decode_ids(self, token_ids: Union[List[int], torch.Tensor, Any]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode. Can be:
                - List[int]: List of token IDs
                - torch.Tensor: PyTorch tensor (1D or 2D)
                - numpy.ndarray: NumPy array (1D or 2D)
                
        Returns:
            Decoded text string (if batched, returns first sequence)
            
        Raises:
            RuntimeError: If tokenizer is not loaded
            ValueError: If input format is invalid
        """
        if not self._is_loaded:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        
        # Use Hugging Face tokenizer if available
        if self._is_hf_tokenizer and self.hf_tokenizer:
            try:
                # Handle tensor conversion
                if isinstance(token_ids, torch.Tensor):
                    if token_ids.dim() > 1:
                        token_ids = token_ids[0]
                    token_ids = token_ids.cpu().tolist()
                elif isinstance(token_ids, list) and len(token_ids) > 0 and isinstance(token_ids[0], list):
                    token_ids = token_ids[0]
                
                return self.hf_tokenizer.decode(token_ids, skip_special_tokens=False)
            except Exception as e:
                logger.error(f"Hugging Face decoding failed: {e}")
                raise RuntimeError(f"Decoding failed: {e}") from e
        
        if not self.sp_processor:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        
        try:
            # Handle PyTorch tensors
            if isinstance(token_ids, torch.Tensor):
                # Convert tensor to list of integers
                token_ids = token_ids.cpu().tolist()
            
            # Handle batched inputs (2D list/array)
            if isinstance(token_ids, list) and len(token_ids) > 0:
                if isinstance(token_ids[0], list):
                    # Batched input, take first sequence
                    token_ids = token_ids[0]
            
            # Handle numpy arrays
            try:
                import numpy as np
                if isinstance(token_ids, np.ndarray):
                    token_ids = token_ids.tolist()
                    # Handle batched numpy arrays
                    if isinstance(token_ids, list) and len(token_ids) > 0:
                        if isinstance(token_ids[0], list):
                            token_ids = token_ids[0]
            except ImportError:
                pass
            
            # Ensure we have a list of integers
            if not isinstance(token_ids, list):
                raise ValueError(f"Expected list of integers, got {type(token_ids)}")
            
            # Convert all elements to integers (in case of mixed types)
            token_ids = [int(tid) for tid in token_ids]
            
            return self.sp_processor.decode_ids(token_ids)
        except Exception as e:
            logger.error(f"Failed to decode token IDs: {e}")
            raise RuntimeError(f"Decoding failed: {e}")
    
    def decode_batch(self, token_ids_batch: Union[List[List[int]], torch.Tensor, Any]) -> List[str]:
        """
        Decode a batch of token ID sequences back to text.
        
        Args:
            token_ids_batch: Batch of token IDs to decode. Can be:
                - List[List[int]]: List of token ID sequences
                - torch.Tensor: PyTorch tensor (2D: [batch_size, seq_len])
                - numpy.ndarray: NumPy array (2D)
                
        Returns:
            List of decoded text strings
            
        Raises:
            RuntimeError: If tokenizer is not loaded
            ValueError: If input format is invalid
        """
        if not self._is_loaded:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        
        # Use Hugging Face tokenizer if available
        if self._is_hf_tokenizer and self.hf_tokenizer:
            try:
                if isinstance(token_ids_batch, torch.Tensor):
                    token_ids_batch = token_ids_batch.cpu().tolist()
                return self.hf_tokenizer.batch_decode(token_ids_batch, skip_special_tokens=False)
            except Exception as e:
                logger.error(f"Hugging Face batch decoding failed: {e}")
                raise RuntimeError(f"Batch decoding failed: {e}") from e
        
        if not self.sp_processor:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        
        try:
            # Handle PyTorch tensors
            if isinstance(token_ids_batch, torch.Tensor):
                # Convert tensor to list of lists
                token_ids_batch = token_ids_batch.cpu().tolist()
            
            # Handle numpy arrays
            try:
                import numpy as np
                if isinstance(token_ids_batch, np.ndarray):
                    token_ids_batch = token_ids_batch.tolist()
            except ImportError:
                pass
            
            # Ensure we have a list of lists
            if not isinstance(token_ids_batch, list):
                raise ValueError(f"Expected list of lists or tensor, got {type(token_ids_batch)}")
            
            # Handle single sequence (1D input)
            if len(token_ids_batch) > 0 and not isinstance(token_ids_batch[0], list):
                # Single sequence provided, wrap in list
                return [self.decode_ids(token_ids_batch)]
            
            # Decode each sequence in the batch
            decoded_texts = []
            for token_ids in token_ids_batch:
                decoded_text = self.decode_ids(token_ids)
                decoded_texts.append(decoded_text)
            
            return decoded_texts
        except Exception as e:
            logger.error(f"Failed to decode token ID batch: {e}")
            raise RuntimeError(f"Batch decoding failed: {e}")
    
    def encode_text(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a single text to token IDs.
        
        Args:
            text: Input text to encode
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            List of token IDs
            
        Raises:
            RuntimeError: If tokenizer is not loaded
        """
        if not self._is_loaded:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        
        # Use Hugging Face tokenizer if available
        if self._is_hf_tokenizer and self.hf_tokenizer:
            try:
                encoded = self.hf_tokenizer.encode(text, add_special_tokens=add_special_tokens)
                return encoded if isinstance(encoded, list) else encoded.tolist()
            except Exception as e:
                logger.error(f"Hugging Face encoding failed: {e}")
                raise RuntimeError(f"Encoding failed: {e}") from e
        
        if not self.sp_processor:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        
        try:
            token_ids = self.sp_processor.encode_as_ids(text)
            
            if add_special_tokens:
                bos_id = self.sp_processor.bos_id()
                eos_id = self.sp_processor.eos_id()
                
                if bos_id != -1:
                    token_ids = [bos_id] + token_ids
                if eos_id != -1:
                    token_ids = token_ids + [eos_id]
            
            return token_ids
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise RuntimeError(f"Encoding failed: {e}")
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        if not self._is_loaded:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        
        if self._is_hf_tokenizer and self.hf_tokenizer:
            return getattr(self.hf_tokenizer, 'vocab_size', len(self.hf_tokenizer.get_vocab()))
        
        if not self.sp_processor:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        return self.sp_processor.get_piece_size()
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size (method version for backward compatibility)."""
        return self.vocab_size
    
    def load_tokenizer(self) -> smp.SentencePieceProcessor:
        """Load tokenizer (alias for load_sentencepiece_tokenizer for consistency)."""
        return self.load_sentencepiece_tokenizer()
    
    @property
    def is_loaded(self) -> bool:
        """Check if tokenizer is loaded."""
        return self._is_loaded
    
    def __repr__(self) -> str:
        """String representation of the TokenizerManager."""
        if self._is_loaded:
            return f"TokenizerManager(loaded=True, vocab_size={self.vocab_size})"
        else:
            return f"TokenizerManager(loaded=False, path='{self.tokenizer_path}')"