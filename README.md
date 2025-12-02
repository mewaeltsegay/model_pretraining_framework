# Model Pretraining Framework

A comprehensive PyTorch-based framework for pretraining and fine-tuning Qwen language models with custom datasets and tokenizers. Optimized for memory efficiency, featuring automatic batch sizing, gradient accumulation, mixed precision training, and comprehensive monitoring.

## 🚀 Features

- **Memory-Aware Training**: Automatic batch size adjustment and gradient accumulation for efficient GPU memory usage
- **Mixed Precision Training**: FP16/BF16 support for faster training and reduced memory footprint
- **Streaming Data Pipeline**: Efficient handling of large datasets without loading everything into memory
- **Custom Tokenizer Support**: Full integration with SentencePiece tokenizers
- **Checkpoint Management**: Automatic checkpointing with best model tracking
- **TensorBoard Integration**: Real-time training metrics visualization
- **Progress Tracking**: Real-time progress bars showing batch-level progress
- **Error Recovery**: Automatic handling of CUDA OOM errors with graceful recovery
- **Configuration Management**: JSON-based configuration with validation
- **Hardware Validation**: Automatic hardware compatibility checks

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mewaeltsegay/model_pretraining_framework.git
   cd model_pretraining_framework
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**:
   - Place your training data in `dataset/train.jsonl`
   - Place validation data in `dataset/validation.jsonl` (optional)
   - Place test data in `dataset/test.jsonl` (optional)
   - Each line should be a JSON object with a `"text"` field

5. **Prepare your tokenizer**:
   - Place your SentencePiece model files in `tokenizer/`
   - Required files:
     - `sentencepiece.model`
     - `sentencepiece.vocab`
     - `tokenizer_config.json`

## 🚀 Quick Start

### Basic Training

1. **Create a configuration file** (or use `config_example.json`):
   ```bash
   cp config_example.json config.json
   ```

2. **Edit `config.json`** with your settings:
   ```json
   {
     "title": "My Training Run",
     "model_name": "Qwen/Qwen2-0.5B",
     "tokenizer_path": "./tokenizer",
     "data_dir": "./dataset",
     "batch_size": 2,
     "gradient_accumulation_steps": 8,
     "max_sequence_length": 512,
     "learning_rate": 5e-5,
     "num_epochs": 3,
     "output_dir": "./checkpoints"
   }
   ```

3. **Start training**:
   ```bash
   python train.py --config config.json
   ```

### Testing the Pipeline

Test your data pipeline and tokenizer:
```bash
python test.py
```

This will:
- Load and tokenize samples from your dataset
- Create DataLoader and process batches
- Load the model and run a forward pass
- Run a few training steps

## 📁 Project Structure

```
model-drive/
├── src/
│   ├── config.py                 # Training configuration dataclass
│   ├── data/
│   │   ├── data_pipeline.py      # Dataset loading and streaming
│   │   └── tokenizer_manager.py  # Tokenizer management
│   ├── models/
│   │   ├── model_manager.py      # Model loading and management
│   │   └── memory_optimizer.py   # Memory optimization utilities
│   ├── training/
│   │   ├── training_controller.py # Main training loop
│   │   └── checkpoint_manager.py # Checkpoint management
│   └── utils/
│       ├── config_utils.py       # Configuration utilities
│       ├── hardware_validator.py # Hardware validation
│       └── logger.py             # Logging setup
├── dataset/                      # Training data (JSONL format)
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── tokenizer/                    # Tokenizer files
│   ├── sentencepiece.model
│   ├── sentencepiece.vocab
│   └── tokenizer_config.json
├── checkpoints/                  # Model checkpoints (auto-created)
├── train.py                      # Main training script
├── test.py                       # Pipeline testing script
├── config_example.json           # Example configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## ⚙️ Configuration

### Configuration File Format

The training configuration can be provided via JSON file or command-line arguments. See `config_example.json` for a complete example.

### Key Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | HuggingFace model identifier | `Qwen/Qwen2-0.5B` |
| `tokenizer_path` | Path to tokenizer directory | `./tokenizer` |
| `data_dir` | Path to dataset directory | `./dataset` |
| `batch_size` | Training batch size | `2` |
| `gradient_accumulation_steps` | Gradient accumulation steps | `8` |
| `max_sequence_length` | Maximum sequence length | `512` |
| `learning_rate` | Learning rate | `5e-5` |
| `num_epochs` | Number of training epochs | `3` |
| `warmup_steps` | Warmup steps for learning rate | `500` |
| `use_mixed_precision` | Enable FP16/BF16 training | `true` |
| `gradient_checkpointing` | Enable gradient checkpointing | `true` |
| `max_gpu_memory_gb` | Maximum GPU memory to use | `5.5` |
| `output_dir` | Checkpoint output directory | `./checkpoints` |
| `enable_tensorboard` | Enable TensorBoard logging | `true` |
| `max_train_samples` | Limit training samples (for testing) | `null` |
| `max_val_samples` | Limit validation samples (for testing) | `null` |

### Command-Line Usage

```bash
# Use JSON config file
python train.py --config config.json

# Override specific parameters
python train.py --config config.json --batch_size 4 --learning_rate 1e-4

# Use default configuration
python train.py
```

## 📊 Training

### Training Process

The training process includes:

1. **Configuration Loading**: Loads and validates configuration from JSON or defaults
2. **Hardware Validation**: Checks GPU availability and memory
3. **Model Loading**: Loads the Qwen model and resizes token embeddings if needed
4. **Data Preparation**: Loads and tokenizes datasets with streaming support
5. **Training Loop**: Runs training with:
   - Automatic batch size adjustment
   - Gradient accumulation
   - Mixed precision training
   - Checkpoint saving
   - Validation evaluation
6. **Monitoring**: Real-time progress bars and TensorBoard logging

### Progress Tracking

During training, you'll see:
- **Progress bars**: Batch-level progress with loss, learning rate, and step number
- **Logging**: Detailed logs saved to `checkpoints/logs/training.log`
- **TensorBoard**: Real-time metrics visualization (if enabled)

### Checkpoints

Checkpoints are automatically saved:
- **Periodic checkpoints**: Every `save_steps` steps
- **Best model**: Saved when validation loss improves
- **Final model**: Saved at the end of training

Checkpoints include:
- Model weights
- Optimizer state
- Scheduler state
- Training configuration
- Training metadata

## 🔍 Monitoring

### TensorBoard

View training metrics in real-time:
```bash
tensorboard --logdir checkpoints/tensorboard_logs
```

Available metrics:
- Training loss
- Validation loss
- Learning rate
- Step time
- GPU memory usage
- Epoch metrics

### Logs

Training logs are saved to:
- `checkpoints/logs/training.log` - Detailed training logs
- Console output - Real-time progress and key metrics

## 🧪 Testing

### Test Data Pipeline

```bash
python test.py
```

This script tests:
1. Tokenizer loading and text tokenization
2. Dataset loading and streaming
3. DataLoader creation and batch processing
4. Model loading and forward pass
5. Training step execution

### Validate Configuration

```bash
python validate_config.py --config config.json
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size`
   - Increase `gradient_accumulation_steps`
   - Reduce `max_sequence_length`
   - Enable `gradient_checkpointing`

2. **Dataset Not Found**
   - Ensure data files exist in `dataset/` directory
   - Check file names: `train.jsonl`, `validation.jsonl`, `test.jsonl`

3. **Tokenizer Loading Errors**
   - Verify tokenizer files exist in `tokenizer/` directory
   - Check `tokenizer_config.json` format

4. **Windows Compatibility**
   - Set `dataloader_num_workers: 0` in config
   - Ensure UTF-8 encoding for console output

### Performance Optimization

- **Memory**: Use gradient accumulation instead of large batch sizes
- **Speed**: Enable mixed precision training (`use_mixed_precision: true`)
- **Data Loading**: Use streaming mode for large datasets
- **Checkpointing**: Adjust `save_steps` based on training duration

## 📚 Dataset Format

The dataset should be in JSONL format (one JSON object per line):

```json
{"text": "Your training text here..."}
{"text": "Another training example..."}
```

Each line must contain:
- A valid JSON object
- A `"text"` field (string, non-empty)
- UTF-8 encoding

See the [Dataset & Tokenizer Documentation](#) for more details.

## 🔧 Development

### Running Tests

```bash
# Test data pipeline
python test.py

# Validate configuration
python validate_config.py --config config.json
```

### Code Structure

- **Modular Design**: Each component is in its own module
- **Type Hints**: Full type annotations for better IDE support
- **Error Handling**: Comprehensive error handling and recovery
- **Logging**: Structured logging throughout

## 📝 License

Please refer to the project license file for usage terms.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📖 Additional Documentation

- **Dataset & Tokenizer**: See the dataset and tokenizer documentation in the README
- **Tokenizer Details**: See `tokenizer/README.md` for tokenizer-specific documentation

## 🙏 Acknowledgments

- Qwen team for the base model
- HuggingFace for the Transformers library
- SentencePiece for tokenization

---

**Last Updated**: 2025-12-02

For questions or issues, please open an issue on the repository.
