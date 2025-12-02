"""
TokenizerManager for handling SentencePiece tokenization in the Qwen pretraining pipeline.

This module provides a unified interface for loading and using the custom SentencePiece
tokenizer with proper error handling and validation.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import torch
import sentencepiece as smp

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
            tokenizer_path: Path to the directory containing tokenizer files
        """
        self.tokenizer_path = Path(tokenizer_path)
        self.sp_processor = None
        self.tokenizer_config = None
        self._is_loaded = False
        
        # Expected tokenizer files
        self.config_file = self.tokenizer_path / "tokenizer_config.json"
        self.model_file = self.tokenizer_path / "sentencepiece.model"
        self.vocab_file = self.tokenizer_path / "sentencepiece.vocab"
        
    def load_sentencepiece_tokenizer(self) -> smp.SentencePieceProcessor:
        """
        Load the SentencePiece tokenizer from local files.
        
        Returns:
            SentencePieceProcessor: Loaded tokenizer processor
            
        Raises:
            FileNotFoundError: If required tokenizer files are missing
            ValueError: If tokenizer files are corrupted or invalid
            RuntimeError: If tokenizer loading fails
        """
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
        if not self._is_loaded or not self.sp_processor:
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
        if not self._is_loaded or not self.sp_processor:
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
        if not self._is_loaded or not self.sp_processor:
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
        if not self._is_loaded or not self.sp_processor:
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
        if not self._is_loaded or not self.sp_processor:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        return self.sp_processor.get_piece_size()
    
    def get_vocab_size(self) -> int:
        """Get the vocabulary size (method version for backward compatibility)."""
        if not self._is_loaded or not self.sp_processor:
            raise RuntimeError("Tokenizer not loaded. Call load_sentencepiece_tokenizer() first.")
        return self.sp_processor.get_piece_size()
    
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