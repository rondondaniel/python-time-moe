# TimeMoE Fine-Tuned Model

## Model Overview

This is a fine-tuned version of the TimeMoE-50M model specifically trained for time series forecasting tasks on the project's dataset. This model has been optimized to improve performance on various forecasting horizons.

**HuggingFace Model**: The model is available on HuggingFace at [rondondaniel/time-moe-webpubs-finetuned](https://huggingface.co/rondondaniel/time-moe-webpubs-finetuned/tree/main).

## Model Details

- **Base Model**: TimeMoE-50M
- **Parameters**: ~50 million
- **Fine-tuning Task**: Time series forecasting
- **Special Capabilities**: Multi-horizon forecasting (1, 7, 30, 60 days)

## Files
The files are available in my HuggingFace model repository at [rondondaniel/time-moe-webpubs-finetuned](https://huggingface.co/rondondaniel/time-moe-webpubs-finetuned/tree/main).

- `model.safetensors`: The model weights (Note: 432.46 MB)
- `config.json`: Model configuration
- `training_args.bin`: Training arguments and hyperparameters used during fine-tuning
- `generation_config.json`: Configuration for text generation

## Fine-tuning Instructions

The TimeMoE-50M model can be fine-tuned on your own dataset for improved performance on specific time series forecasting tasks. Follow these instructions to fine-tune the model:

### Preparing Your Dataset

To start fine-tuning TimeMoE, your dataset should be converted into a jsonl format. Each line represents a time series as a dictionary object with a `sequence` field containing the observations:

```jsonl
{"sequence": [1.0, 2.0, 3.0, ...]}
{"sequence": [11.0, 22.0, 33.0, ...]}
```

Your converted data can be saved in jsonl, json, or pickle format. If you're using the [Time-300B](https://huggingface.co/datasets/Maple728/Time-300B) dataset, no additional preprocessing is needed.

### Training Commands

#### For Small Datasets and small VRAM

If your dataset is small and your VRAM is limited, it's recommended to adjust several parameters for optimal performance. Here's the actual command used for this project's small dataset and small VRAM:

```bash
python main.py -d data/train.jsonl --stride 1 --global-batch-size 2 --precision fp32
```

Parameter explanation:
- `--stride 1`: Ensures all data points are used effectively in a small dataset
- `--global-batch-size 2`: Reduces batch size to reduce VRAM usage
- `--precision fp32`: Uses full precision for more stable training on small datasets

In my environment I only have a GTX 1080 with 8GB of VRAM and CUDA 12.1, running the above command I was able to train the model successfully on my hardware.

#### CPU Training

For training with CPU, execute:

```bash
python main.py -d <data_path>
```

Replace `<data_path>` with the path to your prepared dataset.

#### Single Node with Single or Multiple GPUs

To leverage one or more GPUs on a single node:

```bash
python torch_dist_run.py main.py -d <data_path>
```

#### Multi-Node Multi-GPU Setup

For training across multiple nodes, set up environment variables for inter-node communication:

```bash
export MASTER_ADDR=<master_addr>
export MASTER_PORT=<master_port>
export WORLD_SIZE=<world_size>
export RANK=<rank>
python torch_dist_run.py main.py -d <data_path>
```

#### Training from Scratch

To train TimeMoE from scratch instead of fine-tuning, add the `--from_scratch` argument:

```bash
python torch_dist_run.py main.py -d <data_path> --from_scratch
```

#### Additional Options

Explore additional command-line arguments and their usage:

```bash
python main.py --help
```

## Usage

To use this model for inference:

```python
from transformers import AutoModelForCausalLM
import torch

# Load the model directly from HuggingFace
model_path = "rondondaniel/time-moe-webpubs-finetuned"
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# Prepare input sequence (normalized time series data)
# input_sequence should be a tensor of shape [1, sequence_length]
input_sequence = torch.tensor([...])  # Your preprocessed time series data

# Generate forecast
prediction_length = 30  # Number of steps to forecast
output = model.generate(input_sequence, max_new_tokens=prediction_length)
predictions = output[0, -prediction_length:].cpu().numpy()
```

## Performance

The performance metrics of this model compared to the zero-shot base model can be found in the evaluation metrics file at `4_task4/model_evaluation_metrics.md`. 

Summary of improvements:
- Better performance on short-term forecasts (1-day horizon)
- Competitive performance on medium and long-term forecasts

## Important Notes

1. The model is hosted on HuggingFace, resolving the GitHub file size limitation issues previously encountered.
2. The model requires the same preprocessing steps used during training for optimal results.
3. For best results, normalize input data using the same parameters as during training.
4. When running evaluations with `run_eval.py`, the model is automatically fetched from HuggingFace.

## Citation

If you use this model in your work, please cite:
```
@misc{time-moe-fine-tuned-2025,
  author = {Daniel Rondon},
  title = {Fine-tuned TimeMoE Model for Time Series Forecasting},
  year = {2025},
  publisher = {HuggingFace},
  journal = {HuggingFace Model Hub},
  howpublished = {\url{https://huggingface.co/rondondaniel/time-moe-webpubs-finetuned}}
}
```