#Cullen's incremental Capstone
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
clean_data(df)

# Data type optimization
# Focus on columns such as Temperature, Humidity(%), Wind speed (m/s)
# 'I did all the numerical categories just for practice'
def opt_data(sample):
    df['Rented Bike Count'] = pd.to_numeric(df['Rented Bike Count'], downcast='integer')
    df['Hour'] = pd.to_numeric(df['Hour'], downcast='integer')
    df['Temperature(°C)'] = pd.to_numeric(df['Temperature(°C)'], downcast='float')
    df['Humidity(%)'] = pd.to_numeric(df['Humidity(%)'], downcast='integer')
    df['Wind speed (m/s)'] = pd.to_numeric(df['Wind speed (m/s)'], downcast ='float')
    df['Visibility (10m)'] = pd.to_numeric(df['Visibility (10m)'], downcast ='integer')
    df['Dew point temperature(°C)'] = pd.to_numeric(df['Dew point temperature(°C)'], downcast ='float')
    df['Solar Radiation (MJ/m2)'] = pd.to_numeric(df['Solar Radiation (MJ/m2)'], downcast ='float')
    df['Rainfall(mm)'] = pd.to_numeric(df['Rainfall(mm)'], downcast ='float')
    df['Snowfall (cm)'] = pd.to_numeric(df['Snowfall (cm)'], downcast ='float')
    print("Converted data types for optimization:")
    print(sample.dtypes)
opt_data(df)

# Export the cleaned DataFrame to JSON format
df.to_json('bike_rental_cleaned')

# Write a short report summarizing observations about the data
print('''
      Data has been searched for duplicates and missing values. 
      No replacement or dropped values was needed. 
      Datatypes were downcast to smallest float or integer type for memory optimization''')

#Task #2
## Multiply Temperature by 10 for standardization
df_standard_temp = df['Temperature(°C)'] * 10

# Scale Visibility to a range between 0 and 1 using MinMax scaling
visibility_scaled = df["Visibility (10m)"]
vis_scaled = (visibility_scaled - visibility_scaled.min()) / (visibility_scaled.max() - visibility_scaled.min()) 
print(vis_scaled)

# Conduct basic statistical analysis:
# Use describe() function for key columns like Temperature, Humidity(%), Rented Bike Count
# Compare the results with raw dataset statistics
key_columns = df[['Temperature(°C)', 'Humidity(%)', 'Humidity(%)']]
print(key_columns.describe())

# Identify columns that are not suitable for statistical analysis and recommend possible datetype changes
# # Columns unsuitable for statistical analysis may include non-numeric data types or categorical columns.
# # Recommended changes: Convert such columns to categorical data types for memory optimization.
# Converting categorical data to numerical data
def convert_cat(cat):
    df_func_dumb = pd.get_dummies(df[cat])
    for answer in df_func_dumb:
        count= df_func_dumb[answer].value_counts().astype(int)
        print(f"Number of '{answer}' values in {cat} is {count[True]}")
convert_cat('Functioning Day')
convert_cat('Holiday')
convert_cat('Seasons')

# Export the processed data to a CSV file named bike_rental_processed.csv
df.to_csv("bike_rental_processed.csv") 

# Prepare a short report on statistical observations and insights
print('''
      Temperature can be standardized by multiplying by 10 for better visualization. 
      Visibility was scalled to range between 0 and 1 using MinMax scaling. 
      Descriptive statistics were able to be calculated for Temperature, Humidity, and Rental Bike Count. 
      Categorical data, such as 'Functioning Day', 'Holiday, or 'Seasons' can be converted to numerical data 
      for statistical analysis if desired.''')

# # Task #3
# # Identify categorical and numerical variables
df_categorical = df[['Seasons', 'Holiday', 'Functioning Day']]
df_numerical = df[['Temperature(°C)', 'Humidity(%)', 'Rented Bike Count', 'Wind speed (m/s)', 'Visibility (10m)', 'Dew point temperature(°C)', 'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)']]

# # Perform pivoting operations on the dataset based on categorical columns
# # Group by Seasons and calculate the average rented bike count
df_seasons = df.groupby('Seasons')['Rented Bike Count'].mean().reset_index()
print(df_seasons)

# # Analyze trends across Holiday and Functioning Day
df_hol_fun = df.groupby(['Holiday', 'Functioning Day'])['Rented Bike Count'].mean().reset_index()
print(df_hol_fun)

# # Create distribution tables
# # Temperature and Rented Bike Count distribution by Hour if Hour exists
temp_rent_distribution = df.groupby('Hour')[['Temperature(°C)', 'Rented Bike Count']].mean().reset_index()
print("\nTemperature and Rented Bike Count Distribution by Hour:")
print(temp_rent_distribution)

# # Encode categorical variables and save data as "Rental_Bike_Data_Dummy.csv"
encoded_df = pd.get_dummies(df_categorical, columns=['Seasons', 'Holiday', 'Functioning Day'])
print(encoded_df)
encoded_df.to_csv('Rental_Bike_Data_Dummy.csv')

# # Task 4: Data Visualization
# # Select appropriate visualization techniques for the data:
# # Bar plot for average rentals by Seasons
sns.barplot(x='Seasons', y = 'Rented Bike Count', data=df_seasons)
plt.title("Bar plot: Rentals by Seasons")
plt.savefig("Rented Bike Bar Plot")
plt.show()

# # Line plot showing hourly rentals throughout the day
hour_rent = df.groupby('Hour')['Rented Bike Count'].mean().reset_index()
sns.lineplot(x='Hour', y='Rented Bike Count', data=hour_rent)
plt.title("Line plot: Rentals by hour")
plt.savefig("Rented Bike Line Plot")
plt.show()

# # Heatmap showing correlation among numerical variables
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Variables")
plt.savefig("Heatmap of Numerical Variables")
plt.show()

# # Box plot to identify outliers in Temperature and Rented Bike Count
sns.boxplot(data=df['Rented Bike Count'])
plt.title("Box plot: Rented Bike Count")
plt.savefig("Rented Bike Box Plot")
plt.show()

sns.boxplot(data=df_standard_temp) #standardized temp used for better visualization
plt.title("Box plot: Temperature")
plt.savefig("Temperature Box Plot")
plt.show()

# # Record observations and insights from visualizations
# # Save plots and observations for reporting purposes

print("""
Data Visualization Report:
1. Bar plot shows average bike sales were highest during the Summer season.
2. Line plot shows bike rentals peaked between the hours of 15-20 hour.
3. Heatmap reveals correlations among numerical variables, such as Temperature and Rented Bike Count.
4. Box plots help identify potential outliers in Temperature and Rented Bike Count.
""")
