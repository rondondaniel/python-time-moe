import os
from model_evaluator import TimeMoeEvaluator

if __name__ == "__main__":
    """Main function to evaluate the time series forecasting model."""
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
    model_path_id = "rondondaniel/time-moe-webpubs-finetuned"
    evaluator.load_model(model_path_id)
    for horizon in horizons:
        print(f"\nHorizon: {horizon} days")
        evaluator.evaluate(context_path, evaluation_path, params_path, horizon, target_col)
    
