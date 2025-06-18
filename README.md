# TimeMoE: Time Series Forecasting with Mixture of Experts

## Project Overview

This repository contains a comprehensive implementation of time series forecasting using a Mixture of Experts (MoE) approach. The project provides a complete pipeline for time series analysis, preprocessing, model evaluation, and result visualization. Two models are supported:

- A zero-shot base model (`Maple728/TimeMoE-50M`)
- A fine-tuned model (`rondondaniel/time-moe-webpubs-finetuned`)

The pipeline supports multiple forecast horizons (1, 7, 30, 60 days) and includes tools for data analysis, feature engineering, and performance evaluation.

## Installation and Setup

### Prerequisites

- Python 3.8+
- Conda (for environment management)

### Environment Setup

```bash
# Create and activate the timeseries conda environment
conda create -n timeseries python=3.8
conda activate timeseries

# Install required packages from requirements.txt
pip install -r requirements.txt
```

The project includes a `requirements.txt` file with the following package versions:

- pandas==2.0.3
- numpy==1.24.4
- matplotlib==3.7.4
- seaborn==0.13.1
- statsmodels==0.14.1
- transformers==4.36.2
- scikit-learn==1.3.2
- torch==2.1.2
- tqdm==4.66.1

## Directory Structure

```
python-time-moe/
├── 1_EDA/                     # Exploratory Data Analysis
│   ├── basic_eda.py           # Main EDA script
│   └── plots/                 # Generated EDA plots
├── 2_preprocessing/           # Data preprocessing modules
│   ├── preprocessing.py       # Main preprocessing script
│   ├── time_series_preprocessor.py   # Feature engineering and standardization
│   └── time_series_splitter.py       # Dataset splitting for training/evaluation
├── 3_model_finetuned/         # Fine-tuned model files (if applicable)
├── 4_models_evaluation/       # Model evaluation modules
│   ├── run_eval.py            # Main evaluation script
│   ├── model_evaluator.py     # Model evaluation class
│   └── time_series_forecaster.py     # Forecasting functionality
├── 5_models_comparison/       # Comparison results and visualization
│   ├── forecast plots         # Generated forecast visualizations
│   └── model_evaluation_metrics.md    # Metric comparison tables
├── data/                      # Data files
│   ├── data.csv               # Raw input data
│   ├── processed_data.csv     # Processed data
│   ├── train_data.csv         # Training split
│   ├── context_data.csv       # Context for zero-shot evaluation
│   ├── evaluation_data.csv    # Evaluation for zero-shot
│   ├── context_ft_data.csv    # Context for fine-tuned evaluation
│   ├── evaluation_ft_data.csv # Evaluation for fine-tuned
│   ├── train.jsonl            # Training data in JSONL format for fine-tuning
│   └── preprocessing_params.npy # Saved preprocessing parameters
├── requirements.txt           # Package dependencies with versions
└── README.md                  # This file
```

## Data

The project expects a daily time series dataset with the following structure:
- A date column (named 'day')
- A target column to forecast (named 'target')
- Optional flag features like 'published' and 'is_holiday'
- Other relevant features (if any)

Place your raw data file as `data/data.csv` to use the default paths.

## Modules and Usage

### 1. Exploratory Data Analysis (EDA)

The EDA module provides comprehensive analysis of your time series data.

```bash
# Activate the environment
conda activate timeseries

# Run EDA
cd /path/to/python-time-moe
python 1_EDA/basic_eda.py
```

The script performs:
- Basic statistical analysis
- Target variable analysis by month and day of week
- Time series visualization with seasonal decomposition
- Autocorrelation analysis
- Correlation analysis
- Feature relationship plots
- Distribution analysis

Output is saved to `1_EDA/plots/` and includes console logs with key statistics.

### 2. Data Preprocessing

The preprocessing module handles data transformation and splitting.

```bash
# Run preprocessing
python 2_preprocessing/preprocessing.py
```

Key functionalities:
- Date feature extraction (year, month, day, day of week, etc.)
- Cyclic feature encoding (sine/cosine transformations for month, day of week)
- Feature standardization for continuous variables
- Data splitting into train, context, and evaluation sets for both:
  - Fine-tuning scenario
  - Zero-shot evaluation scenario
- JSONL format conversion for model fine-tuning
- Parameter saving for consistent transformation

The `TimeSeriesPreprocessor` class handles feature engineering and standardization, while the `TimeSeriesSplitter` class manages chronological data splitting.

### 3. Model Fine-tuning

For model fine-tuning, I utilized the code directly from the original TimeMoE repository rather than implementing custom fine-tuning logic in this project. This approach presented several challenges:

- **VRAM Capacity Limitations**: Fine-tuning the large TimeMoE models required significant GPU memory, necessitating a reduction in global batch sizes to accommodate available hardware.

- **Limited Dataset Challenges**: The extremely limited datasets available for fine-tuning. 

- **Limited Documentation**: The codebase was poorly documented, making it difficult to understand their structure and appropriate usage.

- **Early-Stage Codebase**: The fine-tuning and evaluation codebase in the original TimeMoE project appears to be in its initial phase, with limited documentation and examples.

- **Hyperparameter Complexity**: The fine-tuning process involved many hyperparameters to experiment with, including learning rates, batch sizes, gradient accumulation steps, and the number of training epochs.

Despite these challenges, I successfully fine-tuned a model (`rondondaniel/time-moe-webpubs-finetuned`) that showed very slight improvements over the zero-shot model for specific forecast horizons, as demonstrated in the evaluation metrics.

### 4. Model Evaluation

The evaluation module compares model performance across different forecast horizons.

```bash
# Run model evaluation
python 4_models_evaluation/run_eval.py
```

Features:
- Supports evaluation of both zero-shot and fine-tuned models
- Multiple forecast horizons (1, 7, 30, 60 days)
- Metric calculation (MSE, RMSE, MAE)
- Result visualization and saving

The evaluation uses:
- `TimeMoeEvaluator`: Main class for loading models, data, computing metrics, and plotting results
- `TimeSeriesForecaster`: Class for generating forecasts with methods like default and autoregressive

## Results and Evaluation

Model performance is tracked in `5_models_comparison/model_evaluation_metrics.md`, which compares:
- Zero-shot (TimeMoE-50M) performance
- Fine-tuned model performance
- Metrics across different forecast horizons

Visual comparisons of forecasts are available as PNG files in the `5_models_comparison/` directory.

## Key Metrics

The project uses three key metrics to evaluate forecasting performance:
- **MSE (Mean Squared Error)**: Measures average squared differences between predictions and actuals
- **RMSE (Root Mean Squared Error)**: Square root of MSE, more interpretable in original units
- **MAE (Mean Absolute Error)**: Measures average absolute differences without considering direction

These metrics provide complementary insights: MAE measures average magnitude of errors, while RMSE emphasizes larger errors, making it particularly valuable when large forecast deviations are especially problematic in business contexts.

## References

- Base TimeMoE-50M model: [Maple728/TimeMoE-50M](https://huggingface.co/Maple728/TimeMoE-50M)
- Fine-tuned model: [rondondaniel/time-moe-webpubs-finetuned](https://huggingface.co/rondondaniel/time-moe-webpubs-finetuned)
- Hyndman & Athanasopoulos, "Forecasting: Principles and Practice" (2021)
- Sabbha, "Understanding MAE, MSE, and RMSE: Key Metrics in Machine Learning" (2024)

## License

[Specify the license under which the code is distributed]
