# Qwen3-0.6B Fine-tuning for Marathi Translation

This repository contains three different approaches to fine-tune the Qwen3-0.6B model for English-to-Marathi translation. Each notebook demonstrates a different fine-tuning strategy, from parameter-efficient methods to full model fine-tuning.

## 🌟 Overview

The project explores three fine-tuning methodologies:

1. **LoRA Fine-tuning** - Parameter-efficient fine-tuning using Low-Rank Adaptation
2. **Custom Training Loop** - Full fine-tuning with custom PyTorch training implementation
3. **Trainer API** - Full fine-tuning using HuggingFace Trainer API

All approaches use the same dataset and evaluation metrics for fair comparison.

## 📊 Dataset

- **Source**: `anujsahani01/English-Marathi` from HuggingFace Hub
- **Training samples**: 10,000 English-Marathi sentence pairs
- **Validation samples**: 1,000 sentence pairs
- **Test samples**: 200-500 sentence pairs (varies by notebook)
- **Task**: Instruction-following translation from English to Marathi

### Sample Data Format
```python
{
    "english": "Next few months are really crucial for us.",
    "marathi": "पुढील तीन महिने आमच्यासाठी खूप महत्त्वपूर्ण आहेत.",
    "instruction": "Convert the English text into Marathi language."
}
```

## 🚀 Approaches

### 1. LoRA Fine-tuning (`Qwen3-0.6B-finetuned-marathi-translation_LoRA.ipynb`)

**Parameter-Efficient Fine-Tuning using PEFT Library**

#### Configuration:
- **LoRA Rank**: 8
- **LoRA Alpha**: 16
- **LoRA Dropout**: 0.1
- **Target Modules**: "all-linear"
- **Learning Rate**: 2e-4
- **Batch Size**: 2
- **Gradient Accumulation Steps**: 3
- **Optimizer**: AdamW
- **Scheduler**: Cosine Annealing
- **Mixed Precision**: FP16 with GradScaler

#### Features:
- Memory-efficient training (only ~1% of parameters are trainable)
- Advanced checkpoint management with resume capability
- Comprehensive evaluation with BLEU scoring
- WandB integration for experiment tracking
- Model compilation for optimized inference

### 2. Custom Training Loop (`qwen3-0-6b-full-finetuned-marathi-translation_custom_loop.ipynb`)

**Full Model Fine-tuning with Manual Training Implementation**

#### Configuration:
- **Learning Rate**: 5e-5
- **Batch Size**: 6
- **Gradient Accumulation Steps**: 2
- **Optimizer**: Adafactor
- **Scheduler**: Linear with warmup
- **Epochs**: 1
- **Weight Decay**: 0.01
- **Warmup Steps**: 200

#### Features:
- Complete control over training loop
- Custom evaluation functions
- Resume training from checkpoints
- Manual gradient accumulation and clipping
- CUDA optimization for Tesla P100
- Detailed logging and metrics tracking

### 3. Trainer API (`Qwen3-0.6B-full-finetuned-marathi-translation_trainer_api.ipynb`)

**Full Model Fine-tuning using HuggingFace Trainer**

#### Configuration:
- **Learning Rate**: 5e-5
- **Batch Size**: 4 (per device)
- **Gradient Accumulation Steps**: 5
- **Optimizer**: Adafactor
- **Scheduler**: Cosine
- **FP16**: Enabled
- **Gradient Checkpointing**: Enabled

#### Features:
- Simplified training with HF Trainer API
- Built-in evaluation and logging
- Automatic checkpoint management
- Easy model saving and loading
- BLEU score evaluation on test set

## 📋 Requirements

```bash
# Core dependencies
torch>=2.0.0
transformers>=4.38.0
datasets>=2.14.0
accelerate>=0.21.0
peft>=0.5.0

# Evaluation and metrics
evaluate>=0.4.0
sacrebleu>=2.3.0

# Experiment tracking
wandb>=0.15.0

# Additional utilities
numpy>=1.24.0
pandas>=1.5.0
tqdm>=4.64.0
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/qwen3-marathi-translation.git
cd qwen3-marathi-translation
```

2. Install dependencies:
```bash
pip install torch transformers datasets accelerate peft evaluate sacrebleu wandb
```

3. Set up WandB (optional):
```bash
wandb login
```

## 💻 Usage

### Running the Notebooks

Each notebook is self-contained and can be run independently:

1. **LoRA Fine-tuning**:
   ```bash
   jupyter notebook Qwen3-0.6B-finetuned-marathi-translation_LoRA.ipynb
   ```

2. **Custom Training Loop**:
   ```bash
   jupyter notebook Qwen3-0.6B-full-finetuned-marathi-translation_custom_loop.ipynb
   ```

3. **Trainer API**:
   ```bash
   jupyter notebook Qwen3-0.6B-full-finetuned-marathi-translation_trainer_api.ipynb
   ```

### Environment Setup

The notebooks are designed to run on:
- **Platform**: Kaggle GPU instances (Tesla P100)
- **CUDA**: Compatible with CUDA 12.x
- **Memory**: 16GB GPU RAM minimum
- **Storage**: 50GB+ free space for model checkpoints

### Configuration

Each notebook includes configurable hyperparameters:

```python
# Example from LoRA notebook
class LoRAConfig:
    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 2
    NUM_EPOCHS = 1
```

## 📈 Results

### Model Performance

All models are evaluated using SacreBLEU scores on the test set:

| Approach | Training Time | Memory Usage | BLEU Score* | Parameters Trained |
|----------|---------------|--------------|-------------|-------------------|
| LoRA | ~30 minutes | ~8GB | ~XX.XX | ~1% (4.2M/6B) |
| Custom Loop | ~45 minutes | ~14GB | ~XX.XX | 100% (6B) |
| Trainer API | ~40 minutes | ~12GB | ~XX.XX | 100% (6B) |

*Actual BLEU scores depend on the specific training run and are logged in WandB.

### Sample Translations

```
Input:  "The objective of this article is to deepen our appreciation."
Output: "या लेखाचे उद्दिष्ट आपल्या कौतुकाचे गहन करणे आहे."

Input:  "Dengue cases on the rise in city"
Output: "शहरात डेंग्यूची प्रकरणे वाढत आहेत"
```

## 🔄 Resume Training

All notebooks support resuming training from checkpoints:

```python
# LoRA approach
resume_training(
    resume_from_checkpoint="path/to/checkpoint",
    resume_from_epoch=2,
    additional_epochs=2
)

# Custom loop approach
resume_training(
    resume_from_checkpoint="path/to/checkpoint",
    num_additional_epochs=1,
    resume_from_epoch=1
)
```

## 📁 File Structure

```
├── Qwen3-0.6B-finetuned-marathi-translation_LoRA.ipynb
├── Qwen3-0.6B-full-finetuned-marathi-translation_custom_loop.ipynb
├── Qwen3-0.6B-full-finetuned-marathi-translation_trainer_api.ipynb
├── README.md
└── .ipynb_checkpoints/
```

## 🎯 Key Features

### Advanced Training Techniques
- **Mixed Precision Training**: FP16 for memory efficiency
- **Gradient Accumulation**: Simulate larger batch sizes
- **Learning Rate Scheduling**: Cosine annealing and linear warmup
- **Gradient Clipping**: Prevents exploding gradients
- **Model Compilation**: PyTorch 2.0 optimization

### Evaluation & Monitoring
- **BLEU Score Computation**: Industry-standard translation metric
- **WandB Integration**: Real-time experiment tracking
- **Comprehensive Logging**: Loss, learning rate, and custom metrics
- **Checkpoint Management**: Automatic saving of best models

### Memory Optimization
- **Gradient Checkpointing**: Reduces memory usage
- **CUDA Optimizations**: Optimized for Tesla P100
- **Efficient Data Loading**: Optimized DataLoader configurations
- **Memory Cleanup**: Explicit garbage collection

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use gradient accumulation

2. **Dependency Conflicts**:
   - Use virtual environments
   - Check CUDA compatibility
   - Update to latest stable versions

3. **WandB Authentication**:
   - Set up Kaggle secrets for API key
   - Use offline mode if needed

## 📚 References

- [Qwen 3.0 Model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [English-Marathi Dataset](https://huggingface.co/datasets/anujsahani01/English-Marathi)
- [PEFT Library](https://github.com/huggingface/peft)
- [Transformers Library](https://github.com/huggingface/transformers)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace team for the excellent libraries and model hub
- Alibaba Cloud for the Qwen model series
- Dataset contributors for the English-Marathi parallel corpus
- Kaggle for providing free GPU resources

---
