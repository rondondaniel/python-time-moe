import torch
from transformers import AutoModelForCausalLM
import numpy as np


class TimeSeriesForecaster:
    def __init__(self, model: AutoModelForCausalLM, input_sequence: torch.Tensor, prediction_length: int):
        self.model = model
        self.input_sequence = input_sequence
        self.prediction_length = prediction_length

    
    def generate_forecast(self, method='default', **kwargs):
        match method:
            case 'default':
                return self._default_forecast(**kwargs)
            case 'autoregressive':
                return self._autoregressive_forecast()
            case _:
                raise ValueError(f"Unknown method: {method}")
        
    def _default_forecast(self, max_new_tokens=None):
        print(f"Generating forecast for {self.prediction_length} steps ahead...")
        print(f"Using max_new_tokens={max_new_tokens} and default method")
        output = self.model.generate(self.input_sequence, max_new_tokens=max_new_tokens)
        # Extract predictions and ensure correct shape
        predictions = output[0, -self.prediction_length:].cpu().numpy()
        print(f"Predictions shape: {predictions.shape}")
        # Make sure we have the right number of predictions
        predictions = predictions[:min(self.prediction_length, len(predictions))]
        return predictions
    
    def _autoregressive_forecast(self):
        print(f"Generating forecast for {self.prediction_length} steps ahead...")
        print(f"Using autoregressive method")
        with torch.no_grad():
            # Start with input sequence
            working_seq = self.input_sequence.clone()
            predictions = []
            
            # Generate step by step
            for i in range(self.prediction_length):
                # Forward pass to get next time step prediction
                output = self.model(working_seq)
                next_value = output.logits[0, -1]  # Take the last predicted value
                
                # Add to predictions
                predictions.append(next_value.item())
                
                # Update working sequence (remove oldest, add newest)
                working_seq = torch.cat([working_seq[:, 1:], next_value.unsqueeze(0).unsqueeze(1)], dim=1)
                
                # Progress update
                if i % 20 == 0:
                    print(f"  Generated {i}/{self.prediction_length} steps...")
            
            # Convert list to numpy array
            predictions = np.array(predictions)
            return predictions

