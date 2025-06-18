# Time Series Forecasting Model Evaluation Metrics

## Performance Comparison

| **Model** | **Horizon (days)** | **MSE** | **RMSE** | **MAE** |
|-------|----------------|-----|------|-----|
| **Zero-shot (TimeMoE-50M)** | 1 | 418.58 | 20.46 | 20.46 |
| **Zero-shot (TimeMoE-50M)** | 7 | 266,475.77 | 516.21 | 376.37 |
| **Zero-shot (TimeMoE-50M)** | 30 | 76,900.97 | 277.31 | 178.12 |
| **Zero-shot (TimeMoE-50M)** | 60 | 125,788.01 | 354.67 | 236.61 |
| **Fine-tuned model (overfitting)** | 1 | 19,321.01 | 139.00 | 139.00 |
| **Fine-tuned model (overfitting)** | 7 | 201,618.71 | 449.02 | 319.25 |
| **Fine-tuned model (overfitting)** | 30 | 86,547.72 | 294.19 | 233.23 |
| **Fine-tuned model (overfitting)** | 60 | 111,386.43 | 333.75 | 221.53 |
| **Fine-tuned model (second run)** | 1 | 38,895.04 | 197.22 | 197.22 |
| **Fine-tuned model (second run)** | 7 | 175,034.46 | 418.37 | 304.86 |
| **Fine-tuned model (second run)** | 30 | 98,698.19 | 314.16 | 259.66 |
| **Fine-tuned model (second run)** | 60 | 145,938.95 | 382.02 | 286.77 |

## Note

Lower values indicate better performance for all metrics:
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- MAE: Mean Absolute Error

The evaluation was performed using different forecast horizons (1, 7, 30, and 60 days) to test the models' performance on short, medium, and long-term forecasting tasks.

## Metrics selection justification
The selection of Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE) as evaluation metrics for time series forecasting models is well-supported by both theoretical foundations and practical considerations in the field.  
These metrics provide complementary insights: MAE measures average magnitude of errors without considering direction, RMSE emphasizes larger errors, and MSE is mathematically convenient for optimization.  
The combination of these metrics offers a comprehensive assessment framework: MSE (and its square-root variant RMSE) penalizes large prediction errors more heavily, making them particularly valuable when large forecast deviations are especially problematic in business contexts, while MAE provides a more intuitive measurement in the original data units. 