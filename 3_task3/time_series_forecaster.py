import torch
from transformers import AutoModelForCausalLM
import numpy as np


class TimeSeriesForecaster:
    """Time series forecaster."""
    def __init__(self, model: AutoModelForCausalLM, input_sequence: torch.Tensor, prediction_length: int):
        """Initialize the time series forecaster.
        
        Args:
            model (AutoModelForCausalLM): The time series forecasting model.
            input_sequence (torch.Tensor): The input sequence.
            prediction_length (int): The prediction length.
        """
        self.model = model
        self.input_sequence = input_sequence
        self.prediction_length = prediction_length
    
    def generate_forecast(self, method='default', **kwargs): # kwargs for the future
        """Generate a forecast.
        
        Args:
            method (str): The method to use for forecasting.
            **kwargs: Additional keyword arguments.
        
        Returns:
            np.ndarray: The forecast.
        """
        match method:
            case 'default':
                return self._default_forecast()
            case 'autoregressive':
                return self._autoregressive_forecast()
            case _:
                raise ValueError(f"Unknown method: {method}")
        
    def _default_forecast(self):
        """Generate a forecast using the default method."""
        print(f"Generating forecast for {self.prediction_length} steps ahead using default method...")
        output = self.model.generate(self.input_sequence, max_new_tokens=self.prediction_length)
        predictions = output[0, -self.prediction_length:].cpu().numpy()
        predictions = predictions[:min(self.prediction_length, len(predictions))]
        return predictions
    
    def _autoregressive_forecast(self):
        """Generate a forecast using the autoregressive method."""
        print(f"Generating forecast for {self.prediction_length} steps ahead using autoregressive method...")
        with torch.no_grad():
            working_seq = self.input_sequence.clone()
            predictions = []
            try:
                for i in range(self.prediction_length):
                    output = self.model(working_seq)
                    if hasattr(output, 'logits'):
                        next_value = output.logits[0, -1]
                    else:
                        next_value = output[0, -1]
                    predictions.append(next_value.item())
                    working_seq = torch.cat([working_seq[:, 1:], next_value.unsqueeze(0).unsqueeze(1)], dim=1)
                    if i % 20 == 0:
                        print(f"  Generated {i}/{self.prediction_length} steps...")
                print(f"Successfully generated all {self.prediction_length} predictions")
            except Exception as e:
                print(f"Error during autoregressive forecasting: {str(e)}")
                # If we've generated some predictions but encountered an error,
                # we'll use what we have and pad the rest
                if len(predictions) == 0:
                    print("No predictions were generated. Using zeros.")
                    predictions = np.zeros(self.prediction_length)
                    return predictions
                else:
                    print(f"Generated {len(predictions)}/{self.prediction_length} predictions before error.")
                    print(f"Padding remaining predictions with the last valid value.")
            
            # Convert list to numpy array
            predictions = np.array(predictions)
            return predictions

