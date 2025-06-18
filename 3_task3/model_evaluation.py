import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
# Add the directory containing time_series_preprocessor.py to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '1_taks1'))
from time_series_preprocessor import TimeSeriesPreprocessor
from time_series_forecaster import TimeSeriesForecaster

class TimeMoeEvaluator:
    def __init__(self):
        self.model = None
        self.model_name = None

    def load_model(self, model_path_id):
        self.model_name = model_path_id.split('/')[-1]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path_id,
            device_map="cpu",
            trust_remote_code=True,
        )
    
    def _load_preprocessor(self, params_path):
        preprocessor = TimeSeriesPreprocessor()
        preprocessor.load_parameters(params_path)
        return preprocessor

    def _load_data(self,  context_data_path, evaluation_data_path):
        if not os.path.exists(context_data_path):
            raise FileNotFoundError(f"Error: {context_data_path} not found. Make sure to run preprocessing.py first.")

        if not os.path.exists(evaluation_data_path):
            raise FileNotFoundError(f"Error: {evaluation_data_path} not found. Make sure to run preprocessing.py first.")

        context_data = pd.read_csv(context_data_path)
        evaluation_data = pd.read_csv(evaluation_data_path)

        return context_data, evaluation_data

    def _metrics(self, evaluation_data, preprocessor, predictions, target_col, prediction_length):
        actuals = evaluation_data[target_col].values[:prediction_length]
        mean = preprocessor.feature_means[target_col]
        std = preprocessor.feature_stds[target_col]
        original_predictions = predictions * std + mean
        original_actuals = actuals * std + mean

        mse = mean_squared_error(original_actuals, original_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(original_actuals, original_predictions)

        return mse, rmse, mae, original_actuals, original_predictions

    def _plot_results(self, original_actuals, original_predictions, prediction_length):
        model_name = self.model_name.replace('/', '_')
        plt.figure(figsize=(12, 6))
        plt.plot(original_actuals, label='Actual', color='#1f77b4')
        plt.plot(original_predictions, label='Forecast', color='#ff7f0e')
        plt.title(f'{model_name} Time Series Forecast - {prediction_length} days')
        plt.xlabel('Time Steps')
        plt.ylabel('Value') 
        plt.legend()
        plt.grid(True, alpha=0.3)

        os.makedirs('3_task3', exist_ok=True)
        output_path = os.path.join('3_task3', f'forecast_plot_{model_name}_{prediction_length}_days.png')
        plt.savefig(output_path)
        print(f"\nPlot saved to {output_path}")
        plt.close()
    
    def evaluate(self, context_data_path, evaluation_data_path, params_path, prediction_length, target_col, forecast_method='default'):
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

if __name__ == "__main__":
    horizons = [1, 7, 30, 60]
    evaluator = TimeMoeEvaluator()
    target_col = 'target'
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    params_path = os.path.join(project_root, 'data', 'preprocessing_params.npy')

    # Zero-shot evaluation
    print("\nZero-shot evaluation...")
    model_path_id = "Maple728/TimeMoE-50M"
    context_path = os.path.join(project_root, 'data', 'context_data.csv')
    evaluation_path = os.path.join(project_root, 'data', 'evaluation_data.csv')
    evaluator.load_model(model_path_id)
    for horizon in horizons:
        print(f"\nHorizon: {horizon} days")
        evaluator.evaluate(context_path, evaluation_path, params_path, horizon, target_col)

    # Fine-tuned model evaluation
    print("\nFine-tuned model evaluation...")
    model_path_id = os.path.join(project_root, '2_task2', 'model')
    evaluator.load_model(model_path_id)
    for horizon in horizons:
        print(f"\nHorizon: {horizon} days")
        evaluator.evaluate(context_path, evaluation_path, params_path, horizon, target_col)
    








