import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_theme(style="whitegrid")
plt.style.use('seaborn-v0_8')

# Set more readable font size
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Load the data
data_path = "../data/data.csv"

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    # Try alternative path
    data_path = "data/data.csv"
    df = pd.read_csv(data_path)

# Convert date column to datetime
df['day'] = pd.to_datetime(df['day'])

# Set date as index for time series analysis
df_indexed = df.set_index('day')

# Create output directory for plots
os.makedirs('0_task0/plots', exist_ok=True)

#
# 1. Basic Statistical Analysis
# 
print("\n==== 1. BASIC STATISTICAL ANALYSIS ====\n")
print("\nData Overview:")
print(f"Time Range: {df['day'].min()} to {df['day'].max()}")
print(f"Total Observations: {df.shape[0]}")
print(f"Frequency: Daily")
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

df.describe().to_csv('0_task0/statistics_summary.csv')

#
# 2. Target Variable Analysis
#
print("\n==== 2. TARGET VARIABLE ANALYSIS ====\n")

# Monthly aggregation
df['month'] = df['day'].dt.month
df['year'] = df['day'].dt.year
df['day_of_week'] = df['day'].dt.dayofweek
monthly_stats = df.groupby('month')['target'].agg(['mean', 'min', 'max', 'std']).reset_index()
print("\nMonthly Target Statistics:")
print(monthly_stats)

# Daily aggregation
day_of_week_stats = df.groupby('day_of_week')['target'].agg(['mean', 'min', 'max', 'std']).reset_index()
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_of_week_stats['day_name'] = day_of_week_stats['day_of_week'].apply(lambda x: day_names[x])
print("\nDay of Week Target Statistics:")
print(day_of_week_stats)

#
# 3. Time Series Visualization
#
print("\n==== 3. TIME SERIES VISUALIZATION ====\n")

# Plot the target variable over time
fig, ax1 = plt.subplots(figsize=(14, 7))

primary_color = '#1f77b4'    # A nice blue
ax1.set_xlabel('Date')
ax1.set_ylabel('Target', color=primary_color)
ax1.plot(df['day'], df['target'], color=primary_color)
ax1.tick_params(axis='y', labelcolor=primary_color)

ax2 = ax1.twinx()

secondary_color = '#ff7f0e'  # A vibrant orange
ax2.set_ylabel('Published', color=secondary_color)
ax2.plot(df['day'], df['published'], color=secondary_color)
ax2.tick_params(axis='y', labelcolor=secondary_color)

fig.tight_layout()
plt.savefig('0_task0/plots/3_target_time_series.png')

# Plot target by month (boxplot)
plt.figure(figsize=(14, 7))
sns.boxplot(x='month', y='target', data=df)
plt.title('Target Distribution by Month')
plt.xlabel('Month')
plt.ylabel('Target')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('0_task0/plots/3_target_by_month_boxplot.png')

# Plot target by day of week
plt.figure(figsize=(14, 7))
sns.boxplot(x='day_of_week', y='target', data=df)
plt.xticks(range(7), day_names, rotation=45)
plt.title('Target Distribution by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Target')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('0_task0/plots/3_target_by_dow_boxplot.png')

#
# 4. Histogram and Density Plot
#
print("\n==== 4. DISTRIBUTION ANALYSIS ====\n")
plt.figure(figsize=(14, 7))
sns.histplot(df['target'], kde=True, bins=30)
plt.title('Distribution of Target Variable')
plt.xlabel('Target')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('0_task0/plots/4_target_distribution.png')

#
# 5. Seasonal Decomposition
#
print("\n==== 5. SEASONAL DECOMPOSITION ====\n")

# Make sure we have no gaps in dates for decomposition
df_daily = df_indexed.asfreq('D')

try:
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(df_daily['target'], model='additive', period=7)  # Weekly seasonality
    
    # Plot decomposition
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonality')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residuals')
    plt.tight_layout()
    plt.savefig('0_task0/plots/5_seasonal_decomposition_weekly.png')
    
    # Try monthly seasonality if data spans multiple years
    years_span = df['day'].dt.year.max() - df['day'].dt.year.min()
    if years_span >= 1:
        decomposition_monthly = seasonal_decompose(df_daily['target'], model='additive', period=30)  # Monthly seasonality
        
        # Plot monthly decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 12))
        decomposition_monthly.observed.plot(ax=ax1)
        ax1.set_title('Observed')
        decomposition_monthly.trend.plot(ax=ax2)
        ax2.set_title('Trend')
        decomposition_monthly.seasonal.plot(ax=ax3)
        ax3.set_title('Seasonality')
        decomposition_monthly.resid.plot(ax=ax4)
        ax4.set_title('Residuals')
        plt.tight_layout()
        plt.savefig('0_task0/plots/5_seasonal_decomposition_monthly.png')
        print("Created both weekly and monthly seasonal decomposition plots")
    else:
        print("Created weekly seasonal decomposition plot")
        
except Exception as e:
    print(f"Could not perform seasonal decomposition: {e}")

#
# 6. Autocorrelation Analysis
#
print("\n==== 6. AUTOCORRELATION ANALYSIS ====\n")
try:
    # Plot ACF
    plt.figure(figsize=(14, 7))
    plot_acf(df_daily['target'].dropna(), lags=50, alpha=0.05, title='Autocorrelation Function')
    plt.grid(True, alpha=0.3)
    plt.savefig('0_task0/plots/6_acf_plot.png')
    
    # Plot PACF
    plt.figure(figsize=(14, 7))
    plot_pacf(df_daily['target'].dropna(), lags=50, alpha=0.05, title='Partial Autocorrelation Function')
    plt.grid(True, alpha=0.3)
    plt.savefig('0_task0/plots/6_pacf_plot.png')
    print("Created ACF and PACF plots")
    
    # Run Augmented Dickey-Fuller test for stationarity
    result = adfuller(df_daily['target'].dropna())
    print("\nAugmented Dickey-Fuller Test for Stationarity:")
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.4f}")
    
    if result[1] < 0.05:
        print("Result: The series is stationary (reject H0)")
    else:
        print("Result: The series is non-stationary (fail to reject H0)")
        
except Exception as e:
    print(f"Could not perform autocorrelation analysis: {e}")

#
# 7. Correlation Analysis for multivariate data
#
print("\n==== 7. CORRELATION ANALYSIS ====\n")


# Get numeric columns and exclude flag variables
numeric_df = df.select_dtypes(include=[np.number])

# Exclude flag variables from correlation matrix
flag_variables = ['is_holiday', 'published']
numeric_df_no_flags = numeric_df.drop(columns=flag_variables, errors='ignore')

corr_matrix = numeric_df_no_flags.corr()
print("\nCorrelation Matrix (excluding flag variables):")
print(corr_matrix)

# Visualize correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Matrix (excluding flag variables)')
plt.tight_layout()
plt.savefig('0_task0/plots/7_correlation_matrix.png')

#
# 8. Target vs Features
#
print("\n==== 8. TARGET VS FEATURES RELATIONSHIPS ====\n")

# Plot target vs. published
plt.figure(figsize=(14, 7))
sns.boxplot(x='published', y='target', data=df)
plt.title('Target vs. Published')
plt.xlabel('Published')
plt.ylabel('Target')
plt.grid(True, alpha=0.3)
plt.savefig('0_task0/plots/8_target_vs_published.png')

# Target by is_holiday
plt.figure(figsize=(14, 7))
sns.boxplot(x='is_holiday', y='target', data=df)
plt.title('Target Distribution by Holiday Status')
plt.xlabel('Is Holiday (1=Yes, 0=No)')
plt.ylabel('Target')
plt.grid(True, alpha=0.3)
plt.savefig('0_task0/plots/8_target_by_holiday.png')

#
# 9. Published Feature Analysis
#
print("\n==== 9. PUBLISHED & HOLIDAY FEATURES ANALYSIS ====\n")

# Plot published over time
plt.figure(figsize=(14, 7))
plt.plot(df['day'], df['published'], color='#2ca02c', linewidth=1)
plt.title('Published Feature Over Time')
plt.xlabel('Date')
plt.ylabel('Published')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('0_task0/plots/9_published_time_series.png')

plt.figure(figsize=(14, 7))
plt.plot(df['day'], df['is_holiday'], color='#2ca02c', linewidth=1)
plt.title('Holiday Feature Over Time')
plt.xlabel('Date')
plt.ylabel('Holiday')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('0_task0/plots/9_holiday_time_series.png')

#
# 10. Time Series Features
#
print("\n==== 10. TIME SERIES FEATURES ENGINEERING PREVIEW ====\n")

# Create some time-based features for visualization
df['day_of_month'] = df['day'].dt.day
df['quarter'] = df['day'].dt.quarter
df['week_of_year'] = df['day'].dt.isocalendar().week

# Plot target by quarter
plt.figure(figsize=(14, 7))
sns.boxplot(x='quarter', y='target', data=df)
plt.title('Target Distribution by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Target')
plt.grid(True, alpha=0.3)
plt.savefig('0_task0/plots/10_target_by_quarter.png')

# Moving statistics
plt.figure(figsize=(14, 7))
moving_average = df_indexed['target'].rolling(window=7).mean()
moving_std = df_indexed['target'].rolling(window=7).std()
plt.plot(df_indexed.index, df_indexed['target'], label='Original', alpha=0.5)
plt.plot(moving_average.index, moving_average, label='Moving Average (7d)', linewidth=2)
plt.plot(moving_std.index, moving_std, label='Moving STD (7d)', linewidth=2)
plt.title('Moving Statistics (7-day window)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('0_task0/plots/10_moving_statistics.png')

#
# EDA ends
#
print("\nEDA process completed successfully.")
