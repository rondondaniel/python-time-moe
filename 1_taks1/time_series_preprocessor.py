import pandas as pd
import numpy as np
import json
import os

class TimeSeriesPreprocessor:
    """Simplified preprocessor for time series data.
    
    This class handles date feature extraction and standardization of continuous features.
    Flag variables (like published and is_holiday) are kept as-is.
    """
    
    def __init__(self, date_col='day'):
        """Initialize the preprocessor.
        
        Args:
            date_col (str): Name of the date column.
        """
        self.date_col = date_col
        self.continuous_features = ['year', 'month', 'day_num', 'day_of_week', 'quarter', 'target']
        self.flag_features = ['published', 'is_holiday']
        self.feature_means = {}
        self.feature_stds = {}
    
    def _extract_date_features(self, df):
        """Extract features from date column.
        
        Args:
            df (pd.DataFrame): Input DataFrame with date column.
            
        Returns:
            pd.DataFrame: DataFrame with extracted date features.
        """
        df_copy = df.copy()
        
        print(f"Date column '{self.date_col}' dtype before conversion: {df_copy[self.date_col].dtype}")
        
        if not pd.api.types.is_datetime64_any_dtype(df_copy[self.date_col]):
            try:
                df_copy[self.date_col] = pd.to_datetime(df_copy[self.date_col])
                print(f"Successfully converted '{self.date_col}' to datetime")
            except Exception as e:
                print(f"Error converting '{self.date_col}' to datetime: {e}")
                return None
        
        print(f"Date column '{self.date_col}' dtype after conversion: {df_copy[self.date_col].dtype}")
        
        # Extract date features
        df_copy['year'] = df_copy[self.date_col].dt.year
        df_copy['month'] = df_copy[self.date_col].dt.month
        df_copy['day_num'] = df_copy[self.date_col].dt.day  # Rename to avoid confusion with 'day' column
        df_copy['day_of_week'] = df_copy[self.date_col].dt.dayofweek
        df_copy['quarter'] = df_copy[self.date_col].dt.quarter
        
        # Create cyclic features for month and day of week
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        df_copy['day_of_week_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
        df_copy['day_of_week_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
        
        return df_copy
    
    def fit_transform(self, df):
        """Fit the preprocessor and transform the data.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            
        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        df_processed = self._extract_date_features(df)
        
        if df_processed is None:
            print("Error in date feature extraction. Aborting preprocessing.")
            return None
        
        # Scale continuous features
        for feature in self.continuous_features:
            if feature in df_processed.columns:
                # Store mean and std for later use
                self.feature_means[feature] = df_processed[feature].mean()
                self.feature_stds[feature] = df_processed[feature].std()
                
                # Avoid division by zero
                if self.feature_stds[feature] == 0:
                    print(f"Warning: Feature '{feature}' has zero standard deviation. Skipping standardization.")
                    continue
                
                # Apply standardization
                df_processed[feature] = (df_processed[feature] - self.feature_means[feature]) / self.feature_stds[feature]
        
        return df_processed
    
    def transform(self, df):
        """Transform the data using the fitted preprocessor.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            
        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        df_processed = self._extract_date_features(df)
        
        if df_processed is None:
            print("Error in date feature extraction. Aborting transform.")
            return None
        
        # Apply scaling using stored means and stds
        for feature in self.continuous_features:
            if feature in df_processed.columns and feature in self.feature_means:
                # Check if std is not zero to avoid division by zero
                if self.feature_stds.get(feature, 0) != 0:
                    df_processed[feature] = (df_processed[feature] - self.feature_means[feature]) / self.feature_stds[feature]
        
        return df_processed
    
    def save_parameters(self, output_dir='data'):
        """Save preprocessing parameters to a file.
        
        Args:
            output_dir (str): Directory to save parameters.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        params = {
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'continuous_features': self.continuous_features,
            'flag_features': self.flag_features
        }
        
        np.save(f"{output_dir}/preprocessing_params.npy", params)
        print(f"Saved preprocessing parameters to {output_dir}/preprocessing_params.npy")
    
    def load_parameters(self, file_path='data/preprocessing_params.npy'):
        """Load preprocessing parameters from a file.
        
        Args:
            file_path (str): Path to the parameter file.
        """
        params = np.load(file_path, allow_pickle=True).item()
        
        self.feature_means = params['feature_means']
        self.feature_stds = params['feature_stds']
        self.continuous_features = params['continuous_features']
        self.flag_features = params['flag_features']
    
    def to_jsonl(self, df, output_path, column='target'):
        """Convert a pandas DataFrame to a JSONL file format required for fine-tuning time-MOE model.
    
        Args:
            df (pandas.DataFrame): Input DataFrame containing time series data
            output_path (str): Path to save the output JSONL file
        """
        numeric_df = df.select_dtypes(include=['number'])
    
        with open(output_path, 'w') as f:
            for _, row in numeric_df.iterrows():
                json_obj = {"sequence": row.tolist()}
                f.write(json.dumps(json_obj) + '\n')
        