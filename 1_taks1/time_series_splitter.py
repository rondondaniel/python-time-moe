from pandas import DataFrame

class TimeSeriesSplitter:
    def __init__(self, train_ratio=0.7, val_ratio=0.15):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
    
    def split_for_training(self, df: DataFrame):
        return self._train_test_split_time_series(df, self.train_ratio, self.val_ratio)
    
    def split_for_evaluation(self, df: DataFrame):
        return self._context_evaluation_split_time_series(df, self.train_ratio + self.val_ratio)

    def _train_test_split_time_series(self, df: DataFrame, train_ratio=0.7, val_ratio=0.15):
        """Split time series data into train, validation, and test sets chronologically.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            train_ratio (float): Proportion of data to use for training.
            val_ratio (float): Proportion of data to use for validation.
            
        Returns:
            tuple: (train_df, val_df, test_df) containing the split DataFrames.
        """
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # Split chronologically using stratification method
        train_df = df.iloc[:train_size].copy()
        context_df = df.iloc[train_size:train_size+val_size].copy()
        evaluation_df = df.iloc[train_size+val_size:].copy()
        
        train_df.to_csv('data/train_data.csv', index=False)
        context_df.to_csv('data/context_ft_data.csv', index=False)
        evaluation_df.to_csv('data/evaluation_ft_data.csv', index=False)
        
        return train_df, context_df, evaluation_df

    def _context_evaluation_split_time_series(self, df: DataFrame, context_ratio=0.8):
        """Split time series data into train, validation, and test sets chronologically.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            context_ratio (float): Proportion of data to use for context.
            
        Returns:
            tuple: (train_df, val_df, test_df) containing the split DataFrames.
        """
        n = len(df)
        context_size = int(n * context_ratio)
        
        # Split chronologically using stratification method
        context_df = df.iloc[:context_size].copy()
        evaluation_df = df.iloc[context_size:].copy()
        
        context_df.to_csv('data/context_data.csv', index=False)
        evaluation_df.to_csv('data/evaluation_data.csv', index=False)
        
        return context_df, evaluation_df
