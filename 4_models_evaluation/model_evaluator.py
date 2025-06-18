import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
# Add the directory containing time_series_preprocessor.py to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2_preprocessing'))
from time_series_preprocessor import TimeSeriesPreprocessor
from time_series_forecaster import TimeSeriesForecaster

class TimeMoeEvaluator:
    """Evaluator for time series forecasting models."""
    def __init__(self):
        """Initialize the evaluator.
        
        Attributes:
            model (AutoModelForCausalLM): The time series forecasting model.
            model_name (str): The name of the model.
        """
        self.model = None
        self.model_name = None

    def load_model(self, model_path_id):
        """Load the time series forecasting model.
        
        Args:
            model_path_id (str): The path to the model.
        """
        self.model_name = model_path_id.split('/')[-1]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_id,
            device_map="cpu",
            trust_remote_code=True,
        )
    
    def _load_preprocessor(self, params_path):
        """Load the time series preprocessor.
        
        Args:
            params_path (str): The path to the preprocessor parameters.
        """
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_parameters(params_path)
        return preprocessor

    def _load_data(self,  context_data_path, evaluation_data_path):
        """Load the context and evaluation data.
        
        Args:
            context_data_path (str): The path to the context data.
            evaluation_data_path (str): The path to the evaluation data.
        """
        if not os.path.exists(context_data_path):
            raise FileNotFoundError(f"Error: {context_data_path} not found. Make sure to run preprocessing.py first.")
        
        if not os.path.exists(evaluation_data_path):
            raise FileNotFoundError(f"Error: {evaluation_data_path} not found. Make sure to run preprocessing.py first.")
        
        try:
            context_data = pd.read_csv(context_data_path)
            evaluation_data = pd.read_csv(evaluation_data_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
        
        return context_data, evaluation_data

    def _metrics(self, evaluation_data, preprocessor, predictions, target_col, prediction_length):
        """Calculate and plot metrics.
        
        Args:
            evaluation_data (pd.DataFrame): The evaluation data.
            preprocessor (TimeSeriesPreprocessor): The time series preprocessor.
            predictions (np.array): The model predictions.
            target_col (str): The target column.
            prediction_length (int): The prediction length.
        """
        print("\n\nCalculating metrics...")
        
        # Get actual values from evaluation data
        actuals_normalized = evaluation_data[target_col].values[:prediction_length]
        
        # Denormalize to original scale
        target_mean = preprocessor.feature_means[target_col]
        target_std = preprocessor.feature_stds[target_col]
        
        # Denormalize predictions and actuals
        original_predictions = predictions * target_std + target_mean
        original_actuals = actuals_normalized * target_std + target_mean
        
        # Calculate metrics
        mse = mean_squared_error(original_actuals, original_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(original_actuals, original_predictions)
        
        return mse, rmse, mae, original_actuals, original_predictions

    def _plot_results(self, actuals, predictions, prediction_length):
        """Plot the results.
        
        Args:
            actuals (np.array): The actual values.
            predictions (np.array): The predicted values.
            prediction_length (int): The prediction length.
        """
        print("\n\nPlotting results...")
        
        model_name = self.model_name.replace("/", "_") if self.model_name else "model"
        
        plt.figure(figsize=(12, 6))
        plt.plot(actuals, label='Actual', marker='o')
        plt.plot(predictions, label='Predicted', marker='x', linestyle='--')
        plt.title(f'Time Series Forecast for {prediction_length} days using {model_name}')
        plt.xlabel('Days')
        plt.ylabel('Target Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Make sure the directory exists
        os.makedirs('5_models_comparison', exist_ok=True)
        
        # Save the plot
        output_path = os.path.join('5_models_comparison', f'forecast_plot_{model_name}_{prediction_length}_days.png')
        plt.savefig(output_path)
        print(f"\nPlot saved to {output_path}")
        plt.close()
    
    def evaluate(self, context_data_path, evaluation_data_path, params_path, prediction_length, target_col, forecast_method='default'):
        """Evaluate the time series forecasting model.
        
        Args:
            context_data_path (str): The path to the context data.
            evaluation_data_path (str): The path to the evaluation data.
            params_path (str): The path to the preprocessor parameters.
            prediction_length (int): The prediction length.
            target_col (str): The target column.
            forecast_method (str): The forecast method.
        """
        preprocessor = self._load_preprocessor(params_path)
        context_data, evaluation_data = self._load_data(context_data_path, evaluation_data_path)

        context_length = context_data.shape[0]
        target_values = context_data[target_col].values[-context_length:]
        input_sequence = torch.tensor(target_values, dtype=torch.float32).unsqueeze(0)  # [1, context_length]
        
        forecaster = TimeSeriesForecaster(self.model, input_sequence, prediction_length)
        match forecast_method:
            case 'default':
                predictions = forecaster.generate_forecast(method=forecast_method)
            case 'autoregressive':
                predictions = forecaster.generate_forecast(method=forecast_method)
            case _:
                raise ValueError(f"Unknown method or not yet implemented: {forecast_method}")

        mse, rmse, mae, original_actuals, original_predictions = self._metrics(evaluation_data, preprocessor, predictions, target_col, prediction_length)
        print("\nEvaluation Results:")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")

        self._plot_results(original_actuals, original_predictions, prediction_length)