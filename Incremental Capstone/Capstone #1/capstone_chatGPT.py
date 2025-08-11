import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#import dataset
df = pd.read_csv("FloridaBikeRentals.csv", encoding='unicode_escape')

# Inspect the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Dataset shape (number of rows and columns)
print("\nDataset Shape:", df.shape)

# Column names and data types
print("\nColumn Names and Data Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Check for duplicate records
duplicate_count = df.duplicated().sum()
print(f"\nNumber of duplicate records: {duplicate_count}")

# Remove duplicates if any
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New shape: {df.shape}")

# Handle missing values based on column inspection
for col in ['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)']:
    if df[col].isnull().sum() > 0:
        mean_value = df[col].mean()
        print(f"Filling missing values in '{col}' with mean value: {mean_value:.2f}")
        df[col].fillna(mean_value, inplace=True)

# Data type optimizations for memory efficiency
# Suggesting appropriate types for numeric columns
df['Temperature(°C)'] = pd.to_numeric(df['Temperature(°C)'], downcast='float')
df['Humidity(%)'] = pd.to_numeric(df['Humidity(%)'], downcast='integer')
df['Wind speed (m/s)'] = pd.to_numeric(df['Wind speed (m/s)'], downcast='float')

# Verify data type changes
print("\nUpdated Column Data Types:")
print(df.dtypes)

# Export the cleaned DataFrame to JSON format
cleaned_filename = "bike_rental_cleaned.json"
df.to_json(cleaned_filename, orient='records', lines=True)
print(f"\nCleaned data successfully exported to {cleaned_filename}")

# Short report summarizing observations
cleaning_report = """
Data Cleaning Report:
- Initial Shape of Data: {initial_shape}
- Final Shape after Cleaning: {final_shape}
- Duplicate records removed: {duplicate_count}
- Missing values filled with column mean for Temperature, Humidity(%), and Wind speed (m/s)
- Data types optimized for memory efficiency.
Observations:
- Temperature and wind speed are now stored as float types, while Humidity(%) is stored as integer type.
- The dataset is now cleaned and ready for further processing.
""".format(initial_shape=(df.shape[0] + duplicate_count, df.shape[1]), final_shape=df.shape, duplicate_count=duplicate_count)

print("\nData Cleaning Report:")
print(cleaning_report)

# Perform transformations
# Multiply Temperature by 10 for standardization
df['Temperature(°C)'] = df['Temperature(°C)'] * 10

# Scale Visibility to a range between 0 and 1 using MinMax scaling if 'Visibility' exists
if 'Visibility' in df.columns:
    scaler = MinMaxScaler()
    df['Visibility'] = scaler.fit_transform(df[['Visibility']])
    print("\nVisibility column scaled to range [0, 1].")

# Conduct basic statistical analysis
print("\nStatistical Analysis for Key Columns:")
print(df[['Temperature(°C)', 'Humidity(%)', 'Rented Bike Count']].describe())

# Export the processed DataFrame to a CSV file
processed_filename = "bike_rental_processed.csv"
df.to_csv(processed_filename, index=False)
print(f"\nProcessed data successfully exported to {processed_filename}")

# Prepare a short report on statistical observations and insights
statistics_summary = df[['Temperature(°C)', 'Humidity(%)', 'Rented Bike Count']].describe()
processing_report = """
Data Processing and Statistical Analysis Report:
- Temperature values multiplied by 10 for standardization.
- Visibility column scaled to a range between 0 and 1 using MinMax scaling.
- Statistical Summary for Key Columns:
{statistics}
- Columns unsuitable for statistical analysis may include non-numeric data types or categorical columns.
- Recommended changes: Convert such columns to categorical data types for memory optimization.
""".format(statistics=statistics_summary)

print("\nStatistical Analysis Report:")
print(processing_report)

# Task 3: Data Analysis with Pandas

# Identify categorical and numerical variables
categorical_vars = ['Seasons', 'Holiday', 'Functioning Day']
numerical_vars = ['Temperature', 'Humidity(%)', 'Rented Bike Count']
print("\nCategorical Variables:", categorical_vars)
print("Numerical Variables:", numerical_vars)

# Perform pivoting operations
# Group by Seasons and calculate the average rented bike count
season_avg_rentals = df.groupby('Seasons')['Rented Bike Count'].mean().reset_index()
print("\nAverage Rented Bike Count by Seasons:")
print(season_avg_rentals)

# Analyze trends across Holiday and Functioning Day
holiday_trends = df.groupby(['Holiday', 'Functioning Day'])['Rented Bike Count'].mean().reset_index()
print("\nAverage Rented Bike Count by Holiday and Functioning Day:")
print(holiday_trends)

# Create distribution tables
# Temperature and Rented Bike Count distribution by Hour if Hour exists
if 'Hour' in df.columns:
    temp_rent_distribution = df.groupby('Hour')[['Temperature(°C)', 'Rented Bike Count']].mean().reset_index()
    print("\nTemperature and Rented Bike Count Distribution by Hour:")
    print(temp_rent_distribution)

# Seasons and Rented Bike Count distribution
season_rent_distribution = df.groupby('Seasons')['Rented Bike Count'].mean().reset_index()
print("\nSeasons and Rented Bike Count Distribution:")
print(season_rent_distribution)

# Encode categorical variables and save the data to a new CSV
encoded_df = pd.get_dummies(df, columns=categorical_vars, prefix=categorical_vars)
dummy_filename = "Rental_Bike_Data_Dummy.csv"
encoded_df.to_csv(dummy_filename, index=False)
print(f"\nEncoded data successfully exported to {dummy_filename}")
print(encoded_df)

# Task 4: Data Visualization
print("\nTask 4: Data Visualization")

# Bar plot for average rentals by Seasons
plt.figure(figsize=(8, 5))
sns.barplot(data=season_avg_rentals, x='Seasons', y='Rented Bike Count', palette='viridis')
plt.title('Average Rentals by Seasons')
plt.savefig('barplot_seasons.png')
plt.show()

# Line plot showing hourly rentals throughout the day if Hour exists
if 'Hour' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=temp_rent_distribution, x='Hour', y='Rented Bike Count', marker='o', color='blue')
    plt.title('Hourly Rentals Throughout the Day')
    plt.savefig('lineplot_hourly_rentals.png')
    plt.show()

# Heatmap showing correlation among numerical variables
plt.figure(figsize=(8, 6))
correlation_matrix = df[numerical_vars].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap among Numerical Variables')
plt.savefig('heatmap_correlation.png')
plt.show()

# Box plot to identify outliers in Temperature and Rented Bike Count
plt.figure(figsize=(12, 5))
sns.boxplot(data=df[['Temperature(°C)', 'Rented Bike Count']], palette='Set3')
plt.title('Box Plot for Temperature and Rented Bike Count')
plt.savefig('boxplot_temperature_rentals.png')
plt.show()

# Record observations and insights
visualization_report = """
Data Visualization Report:
1. Bar plot shows that average bike rentals vary significantly across seasons.
2. Line plot highlights clear peaks and troughs in hourly rentals throughout the day.
3. Heatmap reveals correlations among numerical variables, such as Temperature and Rented Bike Count.
4. Box plots help identify potential outliers in Temperature and Rented Bike Count.

All plots have been saved for reporting purposes.
"""

print("\nVisualization Report:")
print(visualization_report)
