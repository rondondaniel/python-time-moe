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


# Paths to data files
context_path = os.path.join('data', 'context_data.csv')
evaluation_path = os.path.join('data', 'evaluation_data.csv')
params_path = os.path.join('data', 'preprocessing_params.npy')

# Check if data files exist
if not os.path.exists(context_path):
    print(f"Error: {context_path} not found. Make sure to run preprocessing.py first.")
    sys.exit(1)

if not os.path.exists(evaluation_path):
    print(f"Error: {evaluation_path} not found. Make sure to run preprocessing.py first.")
    sys.exit(1)

# Load preprocessed data
print("Loading preprocessed data...")
context_data = pd.read_csv(context_path)
evaluation_data = pd.read_csv(evaluation_path)

# Filter out columns with 'sin' or 'cos' in their names
# sine_cosine_cols = [col for col in context_data.columns if 'sin' in col.lower() or 'cos' in col.lower()]
# if sine_cosine_cols:
#     print(f"Excluding sine/cosine columns from evaluation: {sine_cosine_cols}")
#     context_data = context_data.drop(columns=sine_cosine_cols)
#     evaluation_data = evaluation_data.drop(columns=sine_cosine_cols)

# Load preprocessing parameters
preprocessor = TimeSeriesPreprocessor()
preprocessor._load_parameters(params_path)
print("Preprocessing parameters loaded.")

# Configuration
context_length = 12  # Default context window for TimeMoE
prediction_length = min(208, len(evaluation_data))  # How many steps to predict
target_col = 'target'  # Column name for the target values

# Create sequences for the model
print(f"Preparing sequences with context_length={context_length}...")
# Get the target values
target_values = context_data[target_col].values[-context_length:]
print(f"Using last {context_length} points from context data for prediction.")

# Reshape as tensor for TimeMoE
input_sequence = torch.tensor(target_values, dtype=torch.float32).unsqueeze(0)  # [1, context_length]
print(f"Input sequence shape: {input_sequence.shape}")

# Load model
print("Loading TimeMoE-50M model...")
model = AutoModelForCausalLM.from_pretrained(
    'Maple728/TimeMoE-50M',
    device_map="cpu",
    trust_remote_code=True,
)

# Generate forecasts using simple autoregressive approach instead of model.generate
print(f"Generating forecast for {prediction_length} steps ahead...")
with torch.no_grad():
    # Start with input sequence
    working_seq = input_sequence.clone()
    predictions = []
    
    # Generate step by step
    for i in range(prediction_length):
        # Forward pass to get next time step prediction
        output = model(working_seq)
        next_value = output.logits[0, -1]  # Take the last predicted value
        
        # Add to predictions
        predictions.append(next_value.item())
        
        # Update working sequence (remove oldest, add newest)
        working_seq = torch.cat([working_seq[:, 1:], next_value.unsqueeze(0).unsqueeze(1)], dim=1)
        
        # Progress update
        if i % 20 == 0:
            print(f"  Generated {i}/{prediction_length} steps...")
    
# Convert list to numpy array
predictions = np.array(predictions)

# Get actual values for comparison
actuals = evaluation_data[target_col].values[:prediction_length]

# Inverse transform predictions and actuals to original scale
mean = preprocessor.feature_means[target_col]
std = preprocessor.feature_stds[target_col]
original_predictions = predictions * std + mean
original_actuals = actuals * std + mean

# Calculate metrics
mse = mean_squared_error(original_actuals, original_predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(original_actuals, original_predictions)

# Print results
print("\nEvaluation Results:")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(original_actuals, label='Actual', color='#1f77b4')
plt.plot(original_predictions, label='Forecast', color='#ff7f0e')
plt.title('TimeMoE Time Series Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Value') 
plt.legend()
plt.grid(True, alpha=0.3)

# Create output directory if it doesn't exist
os.makedirs('3_task3', exist_ok=True)

# Save plot
output_path = os.path.join('3_task3', 'forecast_plot.png')
plt.savefig(output_path)
print(f"\nPlot saved to {output_path}")
plt.close()