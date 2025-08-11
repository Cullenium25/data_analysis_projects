import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#Task#1
#import dataset
df = pd.read_csv("FloridaBikeRentals.csv", encoding='unicode_escape')

# inspect data
def inspect_data(sample):
    print("Sample 15 rows:")
    print(df.sample(15))
    print("First 15 rows:")
    print(df.head(15))
    print("Number of rows, columns:")
    print(df.shape)
    print("Column Names:")
    print(df.columns)
    print("Data types:")
    print(df.dtypes)
inspect_data(df)

# Identify missing values and inconsistencies
def clean_data(sample):
    if df.duplicated().sum() == 0:
        print("No duplicates found in data")
    else: 
        df.drop_duplicates(sample)
        print("Duplicate values have been dropped")
    for col in sample:
        if df[col].isnull().sum() >0:
            mean = df[col].mean()
            df[col].fillna(mean)
            print(f"Missing value filled in {col} with mean value")
        else:
            print(f'No missing values in {col}')
# clean_data(df)

# Data type optimization
def opt_data(sample):
    column = input('Which category? ')
    if pd.api.types.is_integer_dtype(sample[column]):
        sample[column] = pd.to_numeric(sample[column], downcast='integer')
    else: 
        pd.api.types.is_float_dtype(sample[column])
        sample[column] = pd.to_numeric(sample[column], downcast='float')
    # df['Rented Bike Count'] = pd.to_numeric(df['Rented Bike Count'], downcast='integer')
    # df['Hour'] = pd.to_numeric(df['Hour'], downcast='integer')
    # df['Temperature(°C)'] = pd.to_numeric(df['Temperature(°C)'], downcast='float')
    # df['Humidity(%)'] = pd.to_numeric(df['Humidity(%)'], downcast='integer')
    # df['Wind speed (m/s)'] = pd.to_numeric(df['Wind speed (m/s)'], downcast ='float')
    # df['Visibility (10m)'] = pd.to_numeric(df['Visibility (10m)'], downcast ='integer')
    # df['Dew point temperature(°C)'] = pd.to_numeric(df['Dew point temperature(°C)'], downcast ='float')
    # df['Solar Radiation (MJ/m2)'] = pd.to_numeric(df['Solar Radiation (MJ/m2)'], downcast ='float')
    # df['Rainfall(mm)'] = pd.to_numeric(df['Rainfall(mm)'], downcast ='float')
    # df['Snowfall (cm)'] = pd.to_numeric(df['Snowfall (cm)'], downcast ='float')
    print("Converted data types for optimization: ")
    print(sample.dtypes)
# opt_data(df)

# Focus on columns such as Temperature, Humidity(%), Wind speed (m/s)
df_selected = df[['Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)']] 
# print(df_selected)

# Export the cleaned DataFrame to JSON format
df.to_json('bike_rental_cleaned')

#Task #2
## Multiply Temperature by 10 for standardization
df_standard_temp = df['Temperature(°C)'] * 10

#Scale visibility
##	Normalization: Min-Max scaling brings all values into the same range, preventing features with larger ranges from dominating features with smaller ranges in certain algorithms (e.g., distance-based models like KNN).
##	Model Efficiency: Many machine learning models perform better when data is normalized.
##	Improved Interpretability: Values between 0 and 1 are easier to understand and visualize.
visibility = df['Visibility (10m)']
visibility_scaled = (visibility - visibility.min()/ (visibility.max()-visibility.min()))


# Conduct basic statistical analysis:
# Use describe() function for key columns like Temperature, Humidity(%), Humidity(%)
# Compare the results with raw dataset statistics
key_columns = df[['Temperature(°C)', 'Humidity(%)', 'Humidity(%)']]
# print(key_columns.describe())

# Identify columns that are not suitable for statistical analysis and recommend possible datetype changes
# # Columns unsuitable for statistical analysis may include non-numeric data types or categorical columns.
# # Recommended changes: Convert such columns to categorical data types for memory optimization.
# Converting categorical data to numerical data
# df_func_dumb = pd.get_dummies(df["Functioning Day"])
# count= df_func_dumb["Yes"].value_counts().astype(int)

def convert_cat(cat):
    df_func_dumb = pd.get_dummies(df[cat])
    for col in df_func_dumb:
        count= df_func_dumb[col].value_counts().astype(int)
        print(f"Number of '{col}' values in {cat} is {count[True]}")
# convert_cat('Functioning Day')
# convert_cat('Seasons')
# print(f"Number of Functioning days is {count[True]}")
# print(f"Number of Non-functioning days is {count[False]}")

# print("Data for Functioning Day could now be used for statistical analysis if desired")

print(df[df['Rented Bike Count'] == df['Rented Bike Count'].max()])


# Task #3
# Identify categorical and numerical variables


# Perform pivoting operations on the dataset based on categorical columns
# Group by Seasons and calculate the average rented bike count


# Analyze trends across Holiday and Functioning Day


# Create distribution tables


# Encode categorical variables and save data as "Rental_Bike_Data_Dummy.csv"



# Task 4: Data Visualization
# Select appropriate visualization techniques for the data:

# Bar plot for average rentals by Seasons
# sns.barplot(data=df_seasons, x='Seasons', y='Rented Bike Count')
# plt.title('Average Rentals by Seasons')
# plt.savefig('barplot_seasons.png')
# plt.show()

# Line plot showing hourly rentals throughout the day
# sns.lineplot(data=temp_rent_dist, x='Hour', y='Rented Bike Count', marker='o', color='blue')
# plt.title('Hourly Rentals Throughout the Day')
# plt.savefig('lineplot_hourly_rentals.png')
# plt.show()

# Heatmap showing correlation among numerical variables
# correlation_matrix = df[['Temperature(°C)', 'Humidity(%)', 'Rented Bike Count', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Correlation Heatmap among Numerical Variables')
# plt.savefig('heatmap_correlation.png')
# plt.show()

# Box plot to identify outliers in Temperature and Rented Bike Count
# sns.boxplot(data=df[['Temperature(°C)', 'Rented Bike Count']])
# plt.title('Box Plot for Temperature and Rented Bike Count')
# plt.savefig('boxplot_temperature_rentals.png')
# plt.show()

# Record observations and insights from visualizations
# Save plots and observations for reporting purposes

# visualization_report = """
# Data Visualization Report:
# 1. Bar plot shows that average bike rentals vary significantly across seasons.
# 2. Line plot highlights clear peaks and troughs in hourly rentals throughout the day.
# 3. Heatmap reveals correlations among numerical variables, such as Temperature and Rented Bike Count.
# 4. Box plots help identify potential outliers in Temperature and Rented Bike Count.

# All plots have been saved for reporting purposes.
# """

# print("\nVisualization Report:")
# print(visualization_report)