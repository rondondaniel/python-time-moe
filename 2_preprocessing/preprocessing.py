import pandas as pd
from time_series_splitter import TimeSeriesSplitter
from time_series_preprocessor import TimeSeriesPreprocessor


def read_data(data_path):
    """Read the data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        try:
            df = pd.read_csv("data/data.csv")
        except FileNotFoundError:
            print("Error: Could not find data file. Please make sure 'data/data.csv' exists.")
            exit(1)
    
    df['day'] = pd.to_datetime(df['day'])
    
    return df

def main():
    """Main function to preprocess the data."""
    data_path = "data/data.csv"
    
    df = read_data(data_path)
    
    print("\nPreprocessing data...")
    splitter = TimeSeriesSplitter()
    preprocessor = TimeSeriesPreprocessor(date_col='day')
    
    # Preprocess the data
    processed_df = preprocessor.fit_transform(df)
    
    if processed_df is not None:
        preprocessor.save_parameters(output_dir='data')
        processed_df.to_csv('data/processed_data.csv', index=False)
        print(f"Saved processed data to data/processed_data.csv")

        # Split the data into train, context for fine-tuning, and evaluation for fine-tuning sets
        train_df, context_ft_df, evaluation_ft_df = splitter.split_for_training(processed_df, train_ratio=0.7, context_ratio=0.25)
        print(f"Train set: {len(train_df)} samples")
        print(f"Context to evaluate FT set: {len(context_ft_df)} samples")
        print(f"Evaluation to evaluate FT set: {len(evaluation_ft_df)} samples")

        preprocessor.to_jsonl(train_df, 'data/train.jsonl')
        
        # Split the data into context and evaluation sets
        context_df, evaluation_df = splitter.split_for_evaluation(processed_df, context_ratio=0.95)
        print(f"Context to evaluate Zero-shot set: {len(context_df)} samples")
        print(f"Evaluation to evaluate Zero-shot set: {len(evaluation_df)} samples")
        
        print("\nPreprocessing complete!")
        print("Note: For model Fine-Tuning load the processed data from 'data/train.jsonl', 'data/context_ft.csv', 'data/evaluation_ft.csv'")
        print("Note: For model Zero-shot evaluation, load the processed data from 'data/context.csv', 'data/evaluation.csv'")
    else:
        print("\nPreprocessing failed. Please check the errors above.")

if __name__ == "__main__":
    main()