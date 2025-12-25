"""
DataPipeline for handling JSONL dataset loading and streaming in the Qwen pretraining pipeline.

This module provides efficient data loading capabilities with streaming support for large datasets,
proper error handling, and integration with the tokenizer system.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Iterator, Union, Tuple
from pathlib import Path
import warnings

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import Dataset as HFDataset, load_dataset

try:
    from .tokenizer_manager import TokenizerManager
except ImportError:
    # Fallback for when running as script
    from tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class JSONLDataset(IterableDataset):
    """
    Iterable dataset for streaming JSONL files.
    
    This dataset streams data from JSONL files without loading everything into memory,
    making it suitable for large datasets that exceed available RAM.
    """
    
    def __init__(
        self,
        file_path: str,
        tokenizer_manager: TokenizerManager,
        max_length: int = 512,
        skip_errors: bool = True,
        max_samples: Optional[int] = None  # Limit number of samples for testing
    ):
        """
        Initialize the JSONL dataset.
        
        Args:
            file_path: Path to the JSONL file
            tokenizer_manager: TokenizerManager instance for text processing
            max_length: Maximum sequence length for tokenization
            skip_errors: Whether to skip corrupted lines or raise errors
        """
        self.file_path = Path(file_path)
        self.tokenizer_manager = tokenizer_manager
        self.max_length = max_length
        self.skip_errors = skip_errors
        self.max_samples = max_samples  # Limit for testing
        
        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")
        
        # Count total lines for progress tracking
        self._total_lines = self._count_lines()
        
        # Adjust total lines if max_samples is set
        if self.max_samples is not None and self.max_samples > 0:
            self._total_lines = min(self._total_lines, self.max_samples)
            logger.info(f"Initialized JSONL dataset: {self.file_path} ({self._total_lines} samples, limited from {self._count_lines()} for testing)")
        else:
            logger.info(f"Initialized JSONL dataset: {self.file_path} ({self._total_lines} lines)")
    
    def _count_lines(self) -> int:
        """Count total lines in the JSONL file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception as e:
            logger.warning(f"Could not count lines in {self.file_path}: {e}")
            return 0
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate through the dataset, yielding tokenized samples.
        
        Yields:
            Dictionary containing tokenized data with input_ids and attention_mask
        """
        line_number = 0
        processed_count = 0
        error_count = 0
        
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Stop if we've reached max_samples limit
                    if self.max_samples is not None and processed_count >= self.max_samples:
                        break
                        
                    line_number += 1
                    
                    try:
                        # Parse JSON line
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        
                        data = json.loads(line)
                        
                        # Extract text field
                        if 'text' not in data:
                            if not self.skip_errors:
                                raise ValueError(f"Missing 'text' field in line {line_number}")
                            logger.warning(f"Skipping line {line_number}: missing 'text' field")
                            error_count += 1
                            continue
                        
                        text = data['text']
                        if not isinstance(text, str):
                            if not self.skip_errors:
                                raise ValueError(f"'text' field must be string in line {line_number}")
                            logger.warning(f"Skipping line {line_number}: 'text' field is not a string")
                            error_count += 1
                            continue
                        
                        if not text.strip():  # Skip empty text
                            continue
                        
                        # Tokenize the text
                        try:
                            tokenized = self.tokenizer_manager.tokenize_batch(
                                [text],
                                max_length=self.max_length,
                                padding=False,  # We'll handle padding in collator
                                truncation=True,
                                return_tensors="pt",
                                add_special_tokens=True
                            )
                            
                            # Extract single sample from batch
                            sample = {
                                'input_ids': tokenized['input_ids'][0],
                                'attention_mask': tokenized['attention_mask'][0],
                                'text': text  # Keep original text for debugging
                            }
                            
                            processed_count += 1
                            yield sample
                            
                        except Exception as e:
                            if not self.skip_errors:
                                raise RuntimeError(f"Tokenization failed for line {line_number}: {e}")
                            logger.warning(f"Skipping line {line_number}: tokenization failed: {e}")
                            error_count += 1
                            continue
                    
                    except json.JSONDecodeError as e:
                        if not self.skip_errors:
                            raise ValueError(f"Invalid JSON in line {line_number}: {e}")
                        logger.warning(f"Skipping line {line_number}: invalid JSON: {e}")
                        error_count += 1
                        continue
                    
                    except Exception as e:
                        if not self.skip_errors:
                            raise RuntimeError(f"Error processing line {line_number}: {e}")
                        logger.warning(f"Skipping line {line_number}: {e}")
                        error_count += 1
                        continue
        
        except Exception as e:
            logger.error(f"Fatal error reading {self.file_path}: {e}")
            raise
        
        logger.info(
            f"Finished processing {self.file_path}: "
            f"{processed_count} samples processed, {error_count} errors"
        )
    
    def __len__(self) -> int:
        """Return approximate dataset size (total lines)."""
        return self._total_lines


class DataPipeline:
    """
    Main data pipeline for handling JSONL datasets with streaming capabilities.
    
    This class manages loading of train, validation, and test datasets from JSONL files,
    provides streaming capabilities for memory-efficient processing, and integrates
    with the tokenizer system for consistent text processing.
    """
    
    def __init__(
        self,
        data_dir: str = "./dataset",
        tokenizer_manager: Optional[TokenizerManager] = None,
        max_sequence_length: int = 512,
        skip_errors: bool = True,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None
    ):
        """
        Initialize the DataPipeline.
        
        Args:
            data_dir: Directory containing JSONL dataset files, or Hugging Face dataset repository ID
                     (e.g., "username/dataset-name")
            tokenizer_manager: TokenizerManager instance (will create if None)
            max_sequence_length: Maximum sequence length for tokenization
            skip_errors: Whether to skip corrupted data or raise errors
            max_train_samples: Limit training samples for testing (None = use all)
            max_val_samples: Limit validation samples for testing (None = use all)
            max_test_samples: Limit test samples for testing (None = use all)
        """
        self.data_dir_str = data_dir
        self.data_dir = Path(data_dir)
        self.max_sequence_length = max_sequence_length
        self.skip_errors = skip_errors
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        
        # Check if this is a Hugging Face dataset repository ID
        # HF repos typically have format "username/repo-name" and don't exist as local paths
        self._is_hf_dataset = (
            "/" in data_dir and 
            not self.data_dir.exists() and
            not data_dir.startswith("./") and
            not data_dir.startswith("../") and
            not os.path.isabs(data_dir)
        )
        
        # Initialize tokenizer manager if not provided
        if tokenizer_manager is None:
            self.tokenizer_manager = TokenizerManager()
            self.tokenizer_manager.load_sentencepiece_tokenizer()
        else:
            self.tokenizer_manager = tokenizer_manager
        
        # Dataset file paths (only used for local paths)
        if not self._is_hf_dataset:
            self.train_file = self.data_dir / "train.jsonl"
            self.validation_file = self.data_dir / "validation.jsonl"
            self.test_file = self.data_dir / "test.jsonl"
            # Validate data directory
            self._validate_data_directory()
        else:
            # Set dummy paths for Hugging Face datasets (to avoid AttributeError)
            # These won't be used, but prevent errors in methods that check them
            self.train_file = Path("/dummy/train.jsonl")
            self.validation_file = Path("/dummy/validation.jsonl")
            self.test_file = Path("/dummy/test.jsonl")
        
        logger.info(f"Initialized DataPipeline with data_dir: {self.data_dir_str}")
        if self._is_hf_dataset:
            logger.info(f"  → Loading dataset from Hugging Face Hub")
    
    def _validate_data_directory(self) -> None:
        """
        Validate the data directory and check for required files.
        
        Raises:
            FileNotFoundError: If data directory or required files are missing
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Check for at least train.jsonl
        if not self.train_file.exists():
            raise FileNotFoundError(f"Training data file not found: {self.train_file}")
        
        # Log available files
        available_files = []
        for file_path, name in [
            (self.train_file, "train.jsonl"),
            (self.validation_file, "validation.jsonl"),
            (self.test_file, "test.jsonl")
        ]:
            if file_path.exists():
                file_size = file_path.stat().st_size / (1024 * 1024)  # Size in MB
                available_files.append(f"{name} ({file_size:.1f}MB)")
        
        logger.info(f"Available dataset files: {', '.join(available_files)}")
    
    def load_datasets(self, use_streaming: bool = True) -> Dict[str, Union[Dataset, IterableDataset]]:
        """
        Load train, validation, and test datasets from JSONL files or Hugging Face Hub.
        
        Args:
            use_streaming: Whether to use streaming datasets for memory efficiency
            
        Returns:
            Dictionary containing loaded datasets with keys: 'train', 'validation', 'test'
            
        Raises:
            FileNotFoundError: If required dataset files are missing
            RuntimeError: If dataset loading fails
        """
        # Load from Hugging Face if it's a repository ID
        if self._is_hf_dataset:
            return self._load_from_huggingface(use_streaming)
        
        # Otherwise load from local files
        datasets = {}
        
        try:
            # Load training dataset (required)
            logger.info("Loading training dataset...")
            if use_streaming:
                datasets['train'] = JSONLDataset(
                    self.train_file,
                    self.tokenizer_manager,
                    max_length=self.max_sequence_length,
                    skip_errors=self.skip_errors,
                    max_samples=self.max_train_samples
                )
                if self.max_train_samples:
                    logger.info(f"  → Training dataset limited to {self.max_train_samples} samples for testing")
            else:
                datasets['train'] = self._load_non_streaming_dataset(self.train_file)
            
            # Load validation dataset (optional)
            if self.validation_file.exists():
                logger.info("Loading validation dataset...")
                if use_streaming:
                    datasets['validation'] = JSONLDataset(
                        self.validation_file,
                        self.tokenizer_manager,
                        max_length=self.max_sequence_length,
                        skip_errors=self.skip_errors,
                        max_samples=self.max_val_samples
                    )
                    if self.max_val_samples:
                        logger.info(f"  → Validation dataset limited to {self.max_val_samples} samples for testing")
                else:
                    datasets['validation'] = self._load_non_streaming_dataset(self.validation_file)
            else:
                logger.warning("Validation dataset not found, skipping...")
            
            # Load test dataset (optional)
            if self.test_file.exists():
                logger.info("Loading test dataset...")
                if use_streaming:
                    datasets['test'] = JSONLDataset(
                        self.test_file,
                        self.tokenizer_manager,
                        max_length=self.max_sequence_length,
                        skip_errors=self.skip_errors,
                        max_samples=self.max_test_samples
                    )
                    if self.max_test_samples:
                        logger.info(f"  → Test dataset limited to {self.max_test_samples} samples for testing")
                else:
                    datasets['test'] = self._load_non_streaming_dataset(self.test_file)
            else:
                logger.warning("Test dataset not found, skipping...")
            
            logger.info(f"Successfully loaded {len(datasets)} datasets")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise RuntimeError(f"Dataset loading failed: {e}")
    
    def _load_from_huggingface(self, use_streaming: bool = True) -> Dict[str, Union[Dataset, IterableDataset]]:
        """
        Load datasets from Hugging Face Hub.
        
        Args:
            use_streaming: Whether to use streaming datasets for memory efficiency
            
        Returns:
            Dictionary containing loaded datasets
            
        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            logger.info(f"Loading dataset from Hugging Face: {self.data_dir_str}")
            
            # Load dataset from Hugging Face Hub
            if use_streaming:
                hf_dataset = load_dataset(self.data_dir_str, streaming=True)
            else:
                hf_dataset = load_dataset(self.data_dir_str, streaming=False)
            
            datasets = {}
            
            # Convert Hugging Face dataset to our format
            # Check available splits
            if isinstance(hf_dataset, dict):
                # Multiple splits available
                for split_name in ['train', 'validation', 'test', 'val']:
                    if split_name in hf_dataset:
                        # Map 'val' to 'validation' for consistency
                        target_name = 'validation' if split_name == 'val' else split_name
                        logger.info(f"Loading {target_name} split from Hugging Face...")
                        
                        split_dataset = hf_dataset[split_name]
                        
                        # Apply sample limits if specified
                        max_samples = None
                        if target_name == 'train' and self.max_train_samples:
                            max_samples = self.max_train_samples
                        elif target_name == 'validation' and self.max_val_samples:
                            max_samples = self.max_val_samples
                        elif target_name == 'test' and self.max_test_samples:
                            max_samples = self.max_test_samples
                        
                        if max_samples and not use_streaming:
                            split_dataset = split_dataset.select(range(min(max_samples, len(split_dataset))))
                            logger.info(f"  → Limited to {max_samples} samples")
                        
                        # Try to get dataset info for length
                        dataset_info = None
                        try:
                            if hasattr(split_dataset, 'info') and hasattr(split_dataset.info, 'num_rows'):
                                dataset_info = split_dataset.info.num_rows
                            elif hasattr(split_dataset, 'num_rows'):
                                dataset_info = split_dataset.num_rows
                        except Exception:
                            pass
                        
                        # Convert to our tokenized format
                        datasets[target_name] = self._convert_hf_dataset(
                            split_dataset, 
                            use_streaming=use_streaming,
                            max_samples=max_samples if use_streaming else None,
                            dataset_length=dataset_info
                        )
            else:
                # Single split, assume it's training data
                logger.info("Loading single split from Hugging Face (assuming train)...")
                max_samples = self.max_train_samples if use_streaming else None
                
                # Try to get dataset info for length
                dataset_info = None
                try:
                    if hasattr(hf_dataset, 'info') and hasattr(hf_dataset.info, 'num_rows'):
                        dataset_info = hf_dataset.info.num_rows
                    elif hasattr(hf_dataset, 'num_rows'):
                        dataset_info = hf_dataset.num_rows
                except Exception:
                    pass
                
                datasets['train'] = self._convert_hf_dataset(
                    hf_dataset, 
                    use_streaming=use_streaming,
                    max_samples=max_samples,
                    dataset_length=dataset_info
                )
            
            logger.info(f"Successfully loaded {len(datasets)} dataset(s) from Hugging Face")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to load dataset from Hugging Face: {e}")
            raise RuntimeError(f"Failed to load dataset from Hugging Face: {e}") from e
    
    def _convert_hf_dataset(
        self, 
        hf_dataset: Union[HFDataset, Any], 
        use_streaming: bool = True,
        max_samples: Optional[int] = None,
        dataset_length: Optional[int] = None
    ) -> Union[Dataset, IterableDataset]:
        """
        Convert Hugging Face dataset to our tokenized format.
        
        Args:
            hf_dataset: Hugging Face dataset
            use_streaming: Whether to use streaming
            max_samples: Maximum number of samples (for streaming)
            
        Returns:
            Tokenized dataset compatible with our pipeline
        """
        if use_streaming:
            # For streaming, create an iterable dataset wrapper
            class HFStreamingDataset(IterableDataset):
                def __init__(self, hf_dataset, tokenizer_manager, max_length, max_samples, dataset_length=None):
                    self.hf_dataset = hf_dataset
                    self.tokenizer_manager = tokenizer_manager
                    self.max_length = max_length
                    self.max_samples = max_samples
                    self._count = 0
                    
                    # Try to get length from the underlying dataset
                    # For streaming datasets, this might not be available
                    self._length = dataset_length  # Use provided length if available
                    
                    if self._length is None:
                        try:
                            # Some Hugging Face datasets have info.num_rows even in streaming mode
                            if hasattr(hf_dataset, 'info') and hasattr(hf_dataset.info, 'num_rows'):
                                self._length = hf_dataset.info.num_rows
                            elif hasattr(hf_dataset, 'num_rows'):
                                self._length = hf_dataset.num_rows
                            elif hasattr(hf_dataset, '__len__'):
                                # Try to get length directly (might not work for streaming)
                                try:
                                    self._length = len(hf_dataset)
                                except (TypeError, NotImplementedError):
                                    pass
                        except Exception:
                            pass
                    
                    # If max_samples is set, use that as length (override if smaller)
                    if max_samples:
                        if self._length is None:
                            self._length = max_samples
                        else:
                            self._length = min(self._length, max_samples)
                
                def __len__(self):
                    """Return dataset length if available, otherwise raise TypeError."""
                    if self._length is not None:
                        return self._length
                    # For streaming datasets without known length, raise TypeError
                    # This is the standard behavior for IterableDataset
                    raise TypeError(
                        "Cannot determine length of streaming dataset. "
                        "Consider using use_streaming=False or setting max_samples."
                    )
                
                def __iter__(self):
                    self._count = 0  # Reset count for each iteration
                    for item in self.hf_dataset:
                        if self.max_samples and self._count >= self.max_samples:
                            break
                        
                        # Extract text field
                        text = item.get('text', '')
                        if not text or not isinstance(text, str):
                            continue
                        
                        # Tokenize
                        tokenized = self.tokenizer_manager.tokenize_batch(
                            [text],
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=True
                        )
                        
                        self._count += 1
                        yield {
                            'input_ids': tokenized['input_ids'][0],
                            'attention_mask': tokenized['attention_mask'][0],
                            'text': text
                        }
            
            return HFStreamingDataset(
                hf_dataset, 
                self.tokenizer_manager, 
                self.max_sequence_length,
                max_samples,
                dataset_length=dataset_length
            )
        else:
            # For non-streaming, create a regular dataset
            class HFDatasetWrapper(Dataset):
                def __init__(self, hf_dataset, tokenizer_manager, max_length):
                    self.hf_dataset = hf_dataset
                    self.tokenizer_manager = tokenizer_manager
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.hf_dataset)
                
                def __getitem__(self, idx):
                    item = self.hf_dataset[idx]
                    text = item.get('text', '')
                    
                    if not text or not isinstance(text, str):
                        # Return empty sample if text is invalid
                        return {
                            'input_ids': torch.tensor([], dtype=torch.long),
                            'attention_mask': torch.tensor([], dtype=torch.long),
                            'text': ''
                        }
                    
                    # Tokenize
                    tokenized = self.tokenizer_manager.tokenize_batch(
                        [text],
                        max_length=self.max_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=True
                    )
                    
                    return {
                        'input_ids': tokenized['input_ids'][0],
                        'attention_mask': tokenized['attention_mask'][0],
                        'text': text
                    }
            
            return HFDatasetWrapper(hf_dataset, self.tokenizer_manager, self.max_sequence_length)
    
    def _load_non_streaming_dataset(self, file_path: Path) -> Dataset:
        """
        Load a non-streaming dataset (loads all data into memory).
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            PyTorch Dataset containing all data
            
        Note:
            This method is provided for compatibility but not recommended for large datasets
        """
        logger.warning(f"Loading {file_path} into memory (non-streaming mode)")
        
        data = []
        line_number = 0
        error_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_number += 1
                    
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        json_data = json.loads(line)
                        if 'text' in json_data and json_data['text'].strip():
                            data.append(json_data['text'])
                    
                    except Exception as e:
                        if self.skip_errors:
                            error_count += 1
                            logger.warning(f"Skipping line {line_number}: {e}")
                        else:
                            raise
            
            logger.info(f"Loaded {len(data)} samples from {file_path} ({error_count} errors)")
            
            # Create a simple dataset class
            class SimpleDataset(Dataset):
                def __init__(self, texts, tokenizer_manager, max_length):
                    self.texts = texts
                    self.tokenizer_manager = tokenizer_manager
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    text = self.texts[idx]
                    tokenized = self.tokenizer_manager.tokenize_batch(
                        [text],
                        max_length=self.max_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=True
                    )
                    return {
                        'input_ids': tokenized['input_ids'][0],
                        'attention_mask': tokenized['attention_mask'][0],
                        'text': text
                    }
            
            return SimpleDataset(data, self.tokenizer_manager, self.max_sequence_length)
            
        except Exception as e:
            logger.error(f"Failed to load non-streaming dataset from {file_path}: {e}")
            raise
    
    def stream_dataset(self, file_path: str) -> Iterator[Dict[str, str]]:
        """
        Stream raw data from a JSONL file without tokenization.
        
        Args:
            file_path: Path to the JSONL file to stream
            
        Yields:
            Dictionary containing raw data from each line
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Streaming data from {file_path}")
        
        line_number = 0
        processed_count = 0
        error_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_number += 1
                    
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        
                        data = json.loads(line)
                        processed_count += 1
                        yield data
                    
                    except json.JSONDecodeError as e:
                        if self.skip_errors:
                            error_count += 1
                            logger.warning(f"Skipping line {line_number}: invalid JSON: {e}")
                        else:
                            raise ValueError(f"Invalid JSON in line {line_number}: {e}")
                    
                    except Exception as e:
                        if self.skip_errors:
                            error_count += 1
                            logger.warning(f"Skipping line {line_number}: {e}")
                        else:
                            raise RuntimeError(f"Error processing line {line_number}: {e}")
        
        except Exception as e:
            logger.error(f"Fatal error streaming {file_path}: {e}")
            raise
        
        logger.info(
            f"Finished streaming {file_path}: "
            f"{processed_count} samples processed, {error_count} errors"
        )
    
    def get_dataset_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available datasets.
        
        Returns:
            Dictionary containing information about each dataset file
        """
        info = {}
        
        # For Hugging Face datasets, return info about the repository
        if self._is_hf_dataset:
            info['source'] = {
                'type': 'huggingface',
                'repo_id': self.data_dir_str,
                'exists': True
            }
            return info
        
        # For local datasets, check file paths
        for file_path, name in [
            (self.train_file, "train"),
            (self.validation_file, "validation"),
            (self.test_file, "test")
        ]:
            if file_path.exists():
                try:
                    file_size = file_path.stat().st_size
                    line_count = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
                    
                    info[name] = {
                        'file_path': str(file_path),
                        'file_size_mb': file_size / (1024 * 1024),
                        'estimated_samples': line_count,
                        'exists': True
                    }
                except Exception as e:
                    info[name] = {
                        'file_path': str(file_path),
                        'error': str(e),
                        'exists': True
                    }
            else:
                info[name] = {
                    'file_path': str(file_path),
                    'exists': False
                }
        
        return info
    
    def validate_datasets(self) -> Dict[str, bool]:
        """
        Validate all available datasets by checking a few samples.
        
        Returns:
            Dictionary indicating validation status for each dataset
        """
        validation_results = {}
        
        # For Hugging Face datasets, validation happens during loading
        if self._is_hf_dataset:
            logger.info("Skipping file-based validation for Hugging Face dataset")
            # Try to load datasets to validate
            try:
                datasets = self.load_datasets(use_streaming=True)
                for split_name in datasets.keys():
                    validation_results[split_name] = True
                    logger.info(f"Dataset {split_name} validation passed (loaded from Hugging Face)")
            except Exception as e:
                logger.error(f"Dataset validation failed: {e}")
                validation_results['train'] = False
            return validation_results
        
        # For local datasets, validate by checking files
        for file_path, name in [
            (self.train_file, "train"),
            (self.validation_file, "validation"),
            (self.test_file, "test")
        ]:
            if not file_path.exists():
                validation_results[name] = False
                continue
            
            try:
                # Check first few lines
                sample_count = 0
                max_samples = 5
                
                for sample in self.stream_dataset(str(file_path)):
                    if 'text' not in sample:
                        logger.error(f"Missing 'text' field in {name} dataset")
                        validation_results[name] = False
                        break
                    
                    if not isinstance(sample['text'], str):
                        logger.error(f"Invalid 'text' field type in {name} dataset")
                        validation_results[name] = False
                        break
                    
                    sample_count += 1
                    if sample_count >= max_samples:
                        break
                else:
                    validation_results[name] = True
                    logger.info(f"Dataset {name} validation passed ({sample_count} samples checked)")
            
            except Exception as e:
                logger.error(f"Dataset {name} validation failed: {e}")
                validation_results[name] = False
        
        return validation_results


class LanguageModelingCollator:
    """
    Custom data collator for language modeling tasks.
    
    This collator handles dynamic batching with sequence length limits,
    proper padding, and creates labels for causal language modeling.
    """
    
    def __init__(
        self,
        tokenizer_manager: TokenizerManager,
        max_length: int = 512,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = "pt"
    ):
        """
        Initialize the language modeling collator.
        
        Args:
            tokenizer_manager: TokenizerManager instance for padding operations
            max_length: Maximum sequence length for batching
            pad_to_multiple_of: Pad sequences to multiple of this value (for efficiency)
            return_tensors: Format of returned tensors ("pt" for PyTorch)
        """
        self.tokenizer_manager = tokenizer_manager
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        
        # Get padding token ID
        if hasattr(tokenizer_manager, '_is_hf_tokenizer') and tokenizer_manager._is_hf_tokenizer and tokenizer_manager.hf_tokenizer:
            # Hugging Face tokenizer
            self.pad_token_id = tokenizer_manager.hf_tokenizer.pad_token_id
            if self.pad_token_id is None:
                self.pad_token_id = tokenizer_manager.hf_tokenizer.unk_token_id
                if self.pad_token_id is None:
                    self.pad_token_id = 0  # Fallback
                logger.warning("No pad token found, using unk token for padding")
        elif hasattr(tokenizer_manager, 'sp_processor') and tokenizer_manager.sp_processor:
            # SentencePiece tokenizer
            self.pad_token_id = tokenizer_manager.sp_processor.pad_id()
            if self.pad_token_id == -1:  # No pad token, use unk
                self.pad_token_id = tokenizer_manager.sp_processor.unk_id()
                logger.warning("No pad token found, using unk token for padding")
        else:
            raise RuntimeError("TokenizerManager not properly initialized")
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples into a properly padded batch.
        
        Args:
            batch: List of samples, each containing 'input_ids' and 'attention_mask'
            
        Returns:
            Dictionary containing batched and padded tensors:
                - input_ids: Padded input token IDs
                - attention_mask: Attention mask for padded sequences
                - labels: Labels for language modeling (shifted input_ids)
        """
        if not batch:
            raise ValueError("Empty batch provided to collator")
        
        # Extract input_ids and attention_masks
        input_ids_list = []
        attention_masks_list = []
        
        for sample in batch:
            if 'input_ids' not in sample or 'attention_mask' not in sample:
                raise ValueError("Sample missing required keys: 'input_ids' or 'attention_mask'")
            
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']
            
            # Ensure tensors are 1D
            if input_ids.dim() > 1:
                input_ids = input_ids.squeeze()
            if attention_mask.dim() > 1:
                attention_mask = attention_mask.squeeze()
            
            # Apply max length limit
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            
            input_ids_list.append(input_ids)
            attention_masks_list.append(attention_mask)
        
        # Find the maximum length in the batch
        max_len_in_batch = max(len(ids) for ids in input_ids_list)
        
        # Apply pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None:
            max_len_in_batch = ((max_len_in_batch + self.pad_to_multiple_of - 1) 
                               // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Ensure we don't exceed max_length
        max_len_in_batch = min(max_len_in_batch, self.max_length)
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_ids, attention_mask in zip(input_ids_list, attention_masks_list):
            # Calculate padding needed
            padding_length = max_len_in_batch - len(input_ids)
            
            if padding_length > 0:
                # Pad input_ids
                padded_ids = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=input_ids.dtype)
                ])
                
                # Pad attention_mask
                padded_mask = torch.cat([
                    attention_mask,
                    torch.zeros(padding_length, dtype=attention_mask.dtype)
                ])
            else:
                padded_ids = input_ids
                padded_mask = attention_mask
            
            padded_input_ids.append(padded_ids)
            padded_attention_masks.append(padded_mask)
        
        # Stack into batch tensors
        batch_input_ids = torch.stack(padded_input_ids)
        batch_attention_mask = torch.stack(padded_attention_masks)
        
        # Create labels for causal language modeling
        # Labels are the same as input_ids, but shifted by one position
        # We use -100 for padded positions (ignored in loss calculation)
        labels = batch_input_ids.clone()
        labels[batch_attention_mask == 0] = -100  # Ignore padded tokens in loss
        
        result = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask,
            'labels': labels
        }
        
        return result


class DynamicBatchSampler:
    """
    Dynamic batch sampler that groups samples by sequence length for efficient batching.
    
    This sampler helps reduce padding by grouping samples of similar lengths together,
    improving training efficiency and memory usage.
    """
    
    def __init__(
        self,
        dataset: Union[Dataset, IterableDataset],
        batch_size: int,
        max_length: int = 512,
        length_tolerance: int = 32,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize the dynamic batch sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Target batch size
            max_length: Maximum sequence length
            length_tolerance: Group sequences within this length difference
            shuffle: Whether to shuffle the batches
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.length_tolerance = length_tolerance
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # For iterable datasets, we can't pre-compute lengths
        self.is_iterable = isinstance(dataset, IterableDataset)
        
        if not self.is_iterable:
            self._prepare_length_buckets()
    
    def _prepare_length_buckets(self):
        """Prepare length buckets for non-iterable datasets."""
        logger.info("Preparing length buckets for dynamic batching...")
        
        # Group samples by length
        length_buckets = {}
        
        for idx in range(len(self.dataset)):
            try:
                sample = self.dataset[idx]
                seq_length = len(sample['input_ids'])
                
                # Find appropriate bucket
                bucket_key = (seq_length // self.length_tolerance) * self.length_tolerance
                bucket_key = min(bucket_key, self.max_length)
                
                if bucket_key not in length_buckets:
                    length_buckets[bucket_key] = []
                
                length_buckets[bucket_key].append(idx)
                
            except Exception as e:
                logger.warning(f"Skipping sample {idx} in length bucketing: {e}")
                continue
        
        self.length_buckets = length_buckets
        logger.info(f"Created {len(length_buckets)} length buckets")
        
        # Log bucket statistics
        for bucket_length, indices in sorted(length_buckets.items()):
            logger.debug(f"Bucket {bucket_length}: {len(indices)} samples")
    
    def __iter__(self):
        """Generate batches with dynamic length grouping."""
        if self.is_iterable:
            # For iterable datasets, we can't pre-group by length
            # Just yield sequential batches
            batch = []
            for sample in self.dataset:
                batch.append(sample)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            
            # Yield remaining samples if not dropping last
            if batch and not self.drop_last:
                yield batch
        
        else:
            # For regular datasets, use length buckets
            all_batches = []
            
            for bucket_length, indices in self.length_buckets.items():
                if self.shuffle:
                    import random
                    random.shuffle(indices)
                
                # Create batches from this bucket
                for i in range(0, len(indices), self.batch_size):
                    batch_indices = indices[i:i + self.batch_size]
                    
                    if len(batch_indices) == self.batch_size or not self.drop_last:
                        batch = [self.dataset[idx] for idx in batch_indices]
                        all_batches.append(batch)
            
            # Shuffle batches if requested
            if self.shuffle:
                import random
                random.shuffle(all_batches)
            
            for batch in all_batches:
                yield batch


def create_dataloaders(
    datasets: Dict[str, Union[Dataset, IterableDataset]],
    tokenizer_manager: TokenizerManager,
    batch_size: int = 2,
    max_length: int = 512,
    num_workers: int = 0,
    use_dynamic_batching: bool = True,
    pad_to_multiple_of: Optional[int] = None
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all provided datasets with proper collation and batching.
    
    Args:
        datasets: Dictionary of datasets (train, validation, test)
        tokenizer_manager: TokenizerManager instance for collation
        batch_size: Batch size for training
        max_length: Maximum sequence length
        num_workers: Number of worker processes for data loading
        use_dynamic_batching: Whether to use dynamic batching by sequence length
        pad_to_multiple_of: Pad sequences to multiple of this value
        
    Returns:
        Dictionary of DataLoaders for each dataset
    """
    # Create collator
    collator = LanguageModelingCollator(
        tokenizer_manager=tokenizer_manager,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of
    )
    
    dataloaders = {}
    
    for split_name, dataset in datasets.items():
        logger.info(f"Creating DataLoader for {split_name} dataset...")
        
        # Determine if we should shuffle (only for training and non-iterable datasets)
        shuffle = (split_name == 'train') and not isinstance(dataset, IterableDataset)
        
        if use_dynamic_batching and not isinstance(dataset, IterableDataset):
            # Use dynamic batch sampler for regular datasets
            batch_sampler = DynamicBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                max_length=max_length,
                shuffle=shuffle,
                drop_last=(split_name == 'train')  # Drop last only for training
            )
            
            dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collator,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        else:
            # Use standard DataLoader
            # For IterableDataset, don't use shuffle parameter
            if isinstance(dataset, IterableDataset):
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    collate_fn=collator,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )
            else:
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    collate_fn=collator,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available(),
                    drop_last=(split_name == 'train')
                )
        
        dataloaders[split_name] = dataloader
        
        # Log DataLoader info
        if hasattr(dataset, '__len__'):
            num_batches = len(dataset) // batch_size
            if not (split_name == 'train' and len(dataset) % batch_size == 0):
                num_batches += 1
            logger.info(f"{split_name} DataLoader: ~{num_batches} batches")
        else:
            logger.info(f"{split_name} DataLoader: streaming (unknown batch count)")
    
    return dataloaders