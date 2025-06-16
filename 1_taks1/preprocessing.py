import pandas as pd
from time_series_splitter import TimeSeriesSplitter
from time_series_preprocessor import TimeSeriesPreprocessor


def read_data(data_path):
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
    data_path = "../data/data.csv"
    
    df = read_data(data_path)
    
    print("\nPreprocessing data...")
    splitter = TimeSeriesSplitter(train_ratio=0.7, val_ratio=0.15)
    preprocessor = TimeSeriesPreprocessor(date_col='day')
    
    # Preprocess the data
    processed_df = preprocessor.fit_transform(df)
    
    if processed_df is not None:
        preprocessor.save_parameters(output_dir='data')
        processed_df.to_csv('data/processed_data.csv', index=False)
        print(f"Saved processed data to data/processed_data.csv")

        # Split the data into train, validation, and test sets
        train_df, val_df, test_df = splitter.split_for_training(processed_df)
        print(f"Train set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")

        # Split the data into context and evaluation sets
        context_df, evaluation_df = splitter.split_for_evaluation(processed_df)
        print(f"Context set: {len(context_df)} samples")
        print(f"Evaluation set: {len(evaluation_df)} samples")
        
        print("\nPreprocessing complete!")
        print("Note: For model training, load the processed data from 'data/train.csv', 'data/val.csv', 'data/test.csv'")
        print("Note: For model evaluation, load the processed data from 'data/context.csv', 'data/evaluation.csv'")
        print("The preprocessing parameters are saved in 'data/preprocessing_params.npy'")
    else:
        print("\nPreprocessing failed. Please check the errors above.")

if __name__ == "__main__":
    main()
