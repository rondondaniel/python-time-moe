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

- `model.safetensors`: The model weights (Note: 432.46 MB)
- `config.json`: Model configuration
- `training_args.bin`: Training arguments and hyperparameters used during fine-tuning
- `generation_config.json`: Configuration for text generation

## Performance

The performance metrics of this model compared to the zero-shot base model can be found in the evaluation metrics file at `4_task4/model_evaluation_metrics.md`. 

Summary of improvements:
- Better performance on short-term forecasts (1-day horizon)
- Competitive performance on medium and long-term forecasts

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