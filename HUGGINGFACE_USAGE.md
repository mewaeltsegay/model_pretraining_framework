# Using Hugging Face Tokenizer and Dataset

Your training framework now supports loading tokenizers and datasets directly from Hugging Face Hub!

## Quick Start

Simply update your config file to use Hugging Face repository IDs instead of local paths:

```json
{
  "tokenizer_path": "username/your-tokenizer-name",
  "data_dir": "username/your-dataset-name",
  ...
}
```

## Examples

### Example 1: Use Hugging Face Tokenizer and Dataset

```json
{
  "title": "Training with HF Resources",
  "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "tokenizer_path": "mewaeltsegay/tigrinya-tokenizer",
  "data_dir": "mewaeltsegay/tigrinya-dataset",
  "batch_size": 32,
  ...
}
```

### Example 2: Mix Local and Hugging Face

```json
{
  "tokenizer_path": "mewaeltsegay/tigrinya-tokenizer",  // From HF
  "data_dir": "./dataset",  // Local files
  ...
}
```

### Example 3: All Local (Default)

```json
{
  "tokenizer_path": "./tokenizer",  // Local
  "data_dir": "./dataset",  // Local
  ...
}
```

## How It Works

The framework automatically detects if a path is a Hugging Face repository by checking:
- Contains a `/` (e.g., `username/repo-name`)
- Doesn't exist as a local file/directory
- Not a relative path starting with `./` or `../`
- Not an absolute path

If detected as a Hugging Face repository:
- **Tokenizer**: Loaded using `transformers.AutoTokenizer.from_pretrained()`
- **Dataset**: Loaded using `datasets.load_dataset()` with automatic conversion to the training format

## Requirements

Make sure you have the required packages:

```bash
pip install transformers datasets
```

If you need to authenticate (for private repositories):

```bash
huggingface-cli login
```

## Dataset Format

Your Hugging Face dataset should have:
- A `text` field in each sample
- Splits named `train`, `validation`, and/or `test` (or `val` for validation)

Example dataset structure:
```python
{
  "train": [{"text": "Sample 1"}, {"text": "Sample 2"}, ...],
  "validation": [{"text": "Val 1"}, ...],
  "test": [{"text": "Test 1"}, ...]
}
```

## Tokenizer Format

Your Hugging Face tokenizer should be compatible with `transformers.AutoTokenizer`. Common formats:
- SentencePiece tokenizers (`.model` and `.vocab` files)
- BPE tokenizers
- WordPiece tokenizers

The framework automatically handles tokenization using the loaded tokenizer.

## Benefits

✅ **No local storage needed** - Load directly from Hugging Face Hub  
✅ **Version control** - Use specific dataset/tokenizer versions  
✅ **Sharing** - Easy to share resources with collaborators  
✅ **Automatic caching** - Hugging Face libraries cache downloads  
✅ **Streaming support** - Large datasets stream without loading into memory  

## Troubleshooting

### Authentication Errors
If you get authentication errors for private repositories:
```bash
huggingface-cli login
# Or set environment variable:
export HF_TOKEN=your_token_here
```

### Dataset Not Found
- Check the repository ID is correct
- Verify the repository is public or you have access
- Ensure the dataset has the expected splits (`train`, `validation`, `test`)

### Tokenizer Not Compatible
- Ensure the tokenizer is uploaded correctly to Hugging Face
- Check that it's compatible with `AutoTokenizer`
- Verify all required files are present in the repository

## Migration from Local to Hugging Face

1. **Upload your tokenizer** to Hugging Face (if not already done)
2. **Upload your dataset** to Hugging Face (if not already done)
3. **Update your config** with the repository IDs
4. **Run training** - the framework will automatically load from Hugging Face!

No code changes needed - just update the config file paths!

