# U-Transformer: A Hierarchical Transformer Architecture for Efficient Language Modeling

## Overview

U-Transformer is an innovative neural architecture that combines the power of Transformers with hierarchical processing inspired by U-Net architectures. This project explores how multi-scale feature processing can improve language modeling efficiency while maintaining strong performance.

## Key Features

- **Hierarchical Architecture**: Multi-stage encoder-decoder structure with downsampling and upsampling
- **Skip Connections**: Residual connections between encoder and decoder stages for gradient flow
- **Flexible Attention**: Configurable attention head strategies (constant heads vs. constant head dimension)
- **Efficient Processing**: Reduced computational complexity through hierarchical token processing
- **Comprehensive Evaluation**: Multiple architectural variants and ablation studies

## Architecture

The U-Transformer follows a U-shaped architecture with three main components:

### 1. Encoder Path
- **Token Embedding**: Maps input tokens to initial embedding dimension
- **Multi-Stage Processing**: Each stage processes tokens at different resolutions
- **Downsampling**: Reduces sequence length while increasing feature dimension
- **Transformer Blocks**: Standard self-attention and MLP layers at each stage

### 2. Bottleneck
- **Deep Processing**: The narrowest stage with the most transformer blocks
- **Feature Compression**: Captures long-range dependencies efficiently

### 3. Decoder Path
- **Upsampling**: Restores original sequence length
- **Skip Connections**: Fuses encoder features with decoder features
- **Reconstruction**: Generates output logits for next-token prediction

## Model Variants

The project includes several architectural variants:

| Variant | Description | Key Parameters |
|---------|-------------|----------------|
| **Baseline GPT** | Standard transformer baseline | 8 layers, 8 heads, 512 dim |
| **U-Constant Heads** | Fixed number of attention heads | 6 heads across all stages |
| **U-Constant Head Dim** | Fixed head dimension | 64-dim heads, variable count |
| **U-Factor 4** | Aggressive downsampling | 4x downsampling factor |
| **U-No Skips** | Ablation without skip connections | Skip connections disabled |
| **U-Deeper Bottleneck** | Extended bottleneck processing | 4 stages, deeper bottleneck |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd U-Transformer

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets
pip install matplotlib tqdm pyyaml
```

## Usage

### Training a Model

```bash
# Train baseline GPT model
python -m src.utx.train --config configs/baseline_small.yaml

# Train U-Transformer with constant head dimension
python -m src.utx.train --config configs/utransformer_constant_head_dim.yaml

# Train with custom data file
python -m src.utx.train --config configs/baseline_small.yaml --data.file path/to/data.txt
```

### Running All Experiments

```bash
# Run all model variants
python -m src.utx.experiments --data.file path/to/data.txt
```

### Model Analysis

```bash
# Print model sizes and parameter counts
python -m src.utx.print_model_sizes

# Plot training curves
python -m src.utx.plot_logs --log data_from_training/baseline/training_log.csv
```

## Configuration

Models are configured via YAML files in the `configs/` directory. Key parameters include:

```yaml
model_type: u_transformer  # or "gpt"
u:
  stages: [384, 256, 160]           # Embedding dimensions per stage
  blocks_per_stage: [4, 4, 8]      # Transformer blocks per stage
  n_heads_base: 6                   # Base number of attention heads
  head_dim: 64                      # Dimension per attention head
  head_strategy: constant_head_dim  # Head allocation strategy
  downsample_factor: 2              # Downsampling factor between stages
  use_skips: true                   # Enable skip connections
```

## Results

The project includes comprehensive experimental results comparing different architectural variants:

- **Training Curves**: Loss progression for all model variants
- **Parameter Efficiency**: Model size comparisons
- **Performance Metrics**: Validation loss comparisons
- **Ablation Studies**: Impact of skip connections, downsampling factors, and attention strategies

Results are stored in `data_from_training/` with training logs and visualization plots.

## Key Findings

1. **Hierarchical Processing**: U-Transformer achieves competitive performance with reduced computational complexity
2. **Skip Connections**: Critical for maintaining gradient flow and performance
3. **Attention Strategy**: Constant head dimension generally outperforms constant head count
4. **Downsampling Factor**: Moderate downsampling (2x) provides good efficiency-performance trade-off
5. **Bottleneck Depth**: Deeper bottleneck stages improve long-range dependency modeling

## File Structure

```
U-Transformer/
├── configs/                    # Model configuration files
├── data_from_training/        # Experimental results and logs
├── src/utx/                   # Main source code
│   ├── models/               # Model implementations
│   ├── modules/               # Core modules (attention, MLP, etc.)
│   ├── config.py             # Configuration management
│   ├── data.py               # Data loading and preprocessing
│   ├── train.py              # Training loop
│   ├── experiments.py        # Experiment orchestration
│   └── utils.py              # Utility functions
└── README.md                  # This file
```

## Technical Details

### Downsampling/Upsampling
- **Downsampling**: Concatenates adjacent tokens and projects to higher dimension
- **Upsampling**: Projects to higher dimension and reshapes to longer sequence
- **Padding**: Handles sequence length mismatches gracefully

### Attention Mechanisms
- **Multi-Head Attention**: Standard scaled dot-product attention
- **Positional Encoding**: Learned positional embeddings per stage
- **Head Strategies**: Flexible attention head allocation across stages

### Training Features
- **Mixed Precision**: Automatic mixed precision training support
- **Gradient Accumulation**: Memory-efficient training for large batches
- **Learning Rate Scheduling**: Cosine annealing with warmup
- **Early Stopping**: Prevents overfitting with validation monitoring

## Future Work

- **Scaling Studies**: Evaluation on larger datasets and models
- **Architecture Search**: Automated discovery of optimal stage configurations
- **Efficiency Analysis**: Detailed FLOP and memory usage comparisons
- **Task Generalization**: Evaluation on diverse NLP tasks beyond language modeling
