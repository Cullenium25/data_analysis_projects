#Cullen's incremental Capstone
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#Task#1
#import dataset
df = pd.read_csv("FloridaBikeRentals.csv", encoding='unicode_escape')

# Inspect the data:
# View the first few rows, shape, column names, and data types
df_15 = df.sample(15) #random sample of 15 rows
print("Number of Rows, Columns:")
print(df_15.shape) #tuple of rows and columns

# # identify missing values
print("Missing Values:")
print(df_15.isnull().sum())

# # identify duplicate values
print("Number of duplicate Values:")
print(df_15.duplicated().sum())

# Handle missing values and data inconsistencies:
# Report missing values and suggest appropriate handling techniques (e.g., fill with mean, drop rows, etc.)
# Check for duplicate records and remove them if necessary
def clean_data(sample):
    print(sample.isnull().sum())
    print(sample.duplicated().sum())
    for col in sample:
        if df[col].isnull().sum() > 0:
            mean_value = df[col].mean()
            print(f"Filling missing values in '{col}' with mean value: {mean_value:.2f}")
            df[col].fillna(mean_value, inplace=True)
        if df[col].isnull().sum() == 0:
            print(f"No missing values in {col}")
print(clean_data(df_15)) #function to check and fill missing data and drop duplicates

# Comment on data types and suggest optimizations for memory efficiency.
# Focus on columns such as Temperature, Humidity(%), Wind speed (m/s)
df_selected = df_15[['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)']] 
print(df_selected)

# Export the cleaned DataFrame to JSON format
df.to_json("bike_rental_cleaned.json")

# Write a short report summarizing observations about the data
print("Data has been searched for duplicates and missing values. No replacement or dropped values needed")

#Task #2
## Multiply Temperature by 10 for standardization
temp_standard = df_selected['Temperature(°C)'] *10
print(temp_standard)

#Scale visibility
##	Normalization: Min-Max scaling brings all values into the same range, preventing features with larger ranges from dominating features with smaller ranges in certain algorithms (e.g., distance-based models like KNN).
##	Model Efficiency: Many machine learning models perform better when data is normalized.
##	Improved Interpretability: Values between 0 and 1 are easier to understand and visualize.
visibility_scaled = df["Visibility (10m)"]
vis_scaled = (visibility_scaled - visibility_scaled.min()) / (visibility_scaled.max() - visibility_scaled.min()) 
print(vis_scaled)

# Conduct basic statistical analysis:
# Use describe() function for key columns like Temperature, Humidity(%), Rented Bike Count
# Compare the results with raw dataset statistics
df_basic = df[['Temperature(°C)', 'Humidity(%)', 'Rented Bike Count']]
print(df_basic.describe())

# Identify columns that are not suitable for statistical analysis and recommend possible datetype changes
# # Columns unsuitable for statistical analysis may include non-numeric data types or categorical columns.
# # Recommended changes: Convert such columns to categorical data types for memory optimization.
# Converting categorical data to numerical data
df_func_dumb = pd.get_dummies(df["Functioning Day"])
print(df_func_dumb)
true_count= df_func_dumb["Yes"].value_counts().astype(int)
false_count= df_func_dumb["No"].value_counts().astype(int)
print(true_count[True])
print("Data for Functioning Day could now be used for statistical analysis if desired")

# Export the processed data to a CSV file named bike_rental_processed.csv
df.to_csv("bike_rental_processed.csv") 

# Prepare a short report on statistical observations and insights
print("Temperature can be standardized by multiplying by 10 for better visualization. Visibility was scalled to range between 0 and 1 using MinMax scaling. Descriptive statistics were able to be calculated for Temperature, Humidity, and Rental Bike Count. Categorical data, such as 'Functioning Day' can be converted to numerical data for statistical analysis if desired.  ")

# Task #3
# Identify categorical and numerical variables
df_categorical = df[['Seasons', 'Holiday', 'Functioning Day']]
print(df_categorical)
df_numerical = df[['Temperature(°C)', 'Humidity(%)', 'Rented Bike Count', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']]
print(df_numerical)

# Perform pivoting operations on the dataset based on categorical columns
# Group by Seasons and calculate the average rented bike count
df_seasons = df.groupby('Seasons')['Rented Bike Count'].mean().reset_index()
print(df_seasons)

# Analyze trends across Holiday and Functioning Day
df_hol_fun = df.groupby(['Holiday', 'Functioning Day'])['Rented Bike Count'].mean().reset_index()
print(df_hol_fun)

# Create distribution tables
# Temperature and Rented Bike Count distribution by Hour if Hour exists
temp_rent_distribution = df.groupby('Hour')[['Temperature(°C)', 'Rented Bike Count']].mean().reset_index()
print("\nTemperature and Rented Bike Count Distribution by Hour:")
print(temp_rent_distribution)

# # Encode categorical variables and save data as "Rental_Bike_Data_Dummy.csv"
# encoded_df = pd.get_dummies(df_categorical, columns=['Seasons', 'Holiday', 'Functioning Day'])
# print(encoded_df)
# encoded_df.to_csv('encoded_df.csv')

# # Task 4: Data Visualization
# # Select appropriate visualization techniques for the data:
# # Bar plot for average rentals by Seasons
# sns.barplot(x='Seasons', y = 'Rented Bike Count', data=df_seasons)
# plt.title("Bar plot: Rentals by Seasons")
# plt.show()

# # Line plot showing hourly rentals throughout the day
# hour_rent = df.groupby('Hour')['Rented Bike Count'].mean().reset_index()
# sns.lineplot(x='Hour', y='Rented Bike Count', data=hour_rent)
# plt.title("Line plot: Rentals by hour")
# plt.show()

# # Heatmap showing correlation among numerical variables
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Correlation Heatmap of Numerical Variables")
# plt.show()

# # Box plot to identify outliers in Temperature and Rented Bike Count
# sns.boxplot(data=df['Rented Bike Count'])
# plt.title("Box plot: Rented Bike Count")
# plt.savefig("Rented Bike Box Plot")
# plt.show()

# sns.boxplot(data=temp_standard) #standardized temp used for better visualization
# plt.title("Box plot: Temperature")
# plt.savefig("Temperature Box Plot")
# plt.show()

# # Record observations and insights from visualizations
# # Save plots and observations for reporting purposes