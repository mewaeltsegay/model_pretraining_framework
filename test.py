import sys
import io
import logging

# Set UTF-8 encoding for stdout to handle Tigrinya characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Setup minimal logging (suppress verbose library logs)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

from src.data.tokenizer_manager import TokenizerManager
from src.data.data_pipeline import DataPipeline, LanguageModelingCollator
from src.models.model_manager import ModelManager
from src.config import TrainingConfig
from src.training.training_controller import TrainingController
from torch.utils.data import DataLoader
import torch

# Initialize tokenizer
logger.info("Loading tokenizer...")
tokenizer_manager = TokenizerManager(tokenizer_path="./tokenizer")
tokenizer_manager.load_sentencepiece_tokenizer()
logger.info(f"Tokenizer loaded (vocab_size: {tokenizer_manager.vocab_size})")

# Initialize data pipeline
logger.info("Loading dataset...")
data_pipeline = DataPipeline(
    data_dir="./dataset",
    tokenizer_manager=tokenizer_manager,
    max_sequence_length=512
)

# Load datasets
datasets = data_pipeline.load_datasets(use_streaming=True)
logger.info(f"Datasets loaded: {list(datasets.keys())}")

# Process first 5 samples from training dataset
logger.info("Tokenizing first 5 samples from training dataset...")
train_dataset = datasets['train']

for i, sample in enumerate(train_dataset):
    if i >= 5:
        break
    
    text = sample.get('text', '')
    input_ids = sample['input_ids']
    attention_mask = sample['attention_mask']
    
    logger.info(f"\n--- Sample {i+1} ---")
    logger.info(f"Text length: {len(text)} chars")
    logger.info(f"Token count: {len(input_ids)} tokens")
    logger.info(f"Text preview: {text[:100]}...")
    logger.info(f"Input IDs shape: {input_ids.shape}")
    
    # Decode to verify
    decoded = tokenizer_manager.decode_ids(input_ids)
    logger.info(f"Decode match: {decoded == text}")

# Next step: Create DataLoader and show batch processing
logger.info("\n" + "="*60)
logger.info("STEP 2: Creating DataLoader and batch processing...")
logger.info("="*60)

# Create data collator for batching
collator = LanguageModelingCollator(
    tokenizer_manager=tokenizer_manager,
    max_length=512
)

# Create DataLoader
batch_size = 4
dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=collator,
    num_workers=0,  # Windows compatibility
    pin_memory=False
)

logger.info(f"DataLoader created with batch_size={batch_size}")

# Process first 2 batches
logger.info("\nProcessing first 2 batches...")
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 2:
        break
    
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch.get('labels', None)
    
    logger.info(f"\n--- Batch {batch_idx + 1} ---")
    logger.info(f"Input IDs shape: {input_ids.shape}")  # [batch_size, seq_len]
    logger.info(f"Attention mask shape: {attention_mask.shape}")
    logger.info(f"Batch size: {input_ids.shape[0]}")
    logger.info(f"Max sequence length in batch: {input_ids.shape[1]}")
    
    if labels is not None:
        logger.info(f"Labels shape: {labels.shape}")
    
    # Show token count per sample in batch
    token_counts = attention_mask.sum(dim=1).tolist()
    logger.info(f"Token counts per sample: {token_counts}")
    
    # Decode first sample in batch
    first_sample_ids = input_ids[0]
    decoded_first = tokenizer_manager.decode_ids(first_sample_ids)
    logger.info(f"First sample decoded length: {len(decoded_first)} chars")

# Next step: Load model and run forward pass
logger.info("\n" + "="*60)
logger.info("STEP 3: Loading model and testing forward pass...")
logger.info("="*60)

# Create minimal config for model loading (skip validation for testing)
try:
    # Temporarily disable CUDA validation
    import os
    original_cuda_check = torch.cuda.is_available
    
    # Create config without validation
    config_dict = {
        "model_name": "Qwen/Qwen2-0.5B",
        "tokenizer_path": "./tokenizer",
        "data_dir": "./dataset",
        "max_sequence_length": 512,
        "use_mixed_precision": False,
        "gradient_checkpointing": False,
        "output_dir": "./test_output",
        "batch_size": 2
    }
    
    # Create config from dict to skip validation
    config = TrainingConfig.from_dict(config_dict)
    # Override validation to pass for testing
    config.validate = lambda: None
    config._validate_hardware_compatibility = lambda: None
except Exception as e:
    logger.warning(f"Config creation warning: {e}, creating minimal config...")
    # Fallback: create config from dict without validation
    config_dict = {
        "model_name": "Qwen/Qwen2-0.5B",
        "tokenizer_path": "./tokenizer",
        "data_dir": "./dataset",
        "max_sequence_length": 512,
        "use_mixed_precision": False,
        "gradient_checkpointing": False,
        "output_dir": "./test_output",
        "batch_size": 2,
        "max_gpu_memory_gb": 5.5
    }
    config = TrainingConfig.from_dict(config_dict)
    config.validate = lambda: None
    config._validate_hardware_compatibility = lambda: None

# Initialize model manager
logger.info("Initializing ModelManager...")
model_manager = ModelManager(config)

# Load model
logger.info("Loading Qwen model...")
model = model_manager.load_qwen_model()

# Get model info
model_info = model_manager.get_model_info()
logger.info(f"Model loaded: {model_info['num_parameters']:,} parameters")
logger.info(f"Model dtype: {model_info['model_dtype']}")
logger.info(f"Device: {model_info['device']}")

# Validate model compatibility with tokenizer
model_manager.validate_model_compatibility(tokenizer_manager.vocab_size)
logger.info("Model-tokenizer compatibility validated")

# Run forward pass with one batch
logger.info("\nRunning forward pass with one batch...")
model.eval()  # Set to eval mode for inference

with torch.no_grad():
    # Get first batch from dataloader
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(model_info['device'])
    attention_mask = batch['attention_mask'].to(model_info['device'])
    labels = batch.get('labels', None)
    if labels is not None:
        labels = labels.to(model_info['device'])
    
    logger.info(f"Batch input shape: {input_ids.shape}")
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels if labels is not None else input_ids
    )
    
    logits = outputs.logits
    loss = outputs.loss if hasattr(outputs, 'loss') and outputs.loss is not None else None
    
    logger.info(f"Logits shape: {logits.shape}")  # [batch_size, seq_len, vocab_size]
    logger.info(f"Loss: {loss.item():.4f}" if loss is not None else "Loss: N/A")
    
    # Show predictions for first token in first sample
    first_token_logits = logits[0, 0, :]  # First sample, first position
    top_5_probs, top_5_indices = torch.topk(torch.softmax(first_token_logits, dim=-1), k=5)
    
    logger.info("\nTop 5 predicted tokens for first position:")
    for i, (prob, idx) in enumerate(zip(top_5_probs, top_5_indices)):
        token = tokenizer_manager.sp_processor.id_to_piece(int(idx))
        logger.info(f"  {i+1}. Token '{token}' (ID: {idx}): {prob:.4f}")

# Next step: Run a few training steps using TrainingController
logger.info("\n" + "="*60)
logger.info("STEP 4: Running training steps using TrainingController...")
logger.info("="*60)

# Update config for training
config.num_epochs = 1  # Just for testing
config.warmup_steps = 2
config.logging_steps = 1  # Log every step for testing

from train import TrainingOrchestrator
training_orchestrator = TrainingOrchestrator(config)
training_orchestrator.run_training()



logger.info("\n✅ Full pipeline test completed using codebase components!")
logger.info("   Pipeline: dataset → tokenization → batching → model → TrainingController → training steps")