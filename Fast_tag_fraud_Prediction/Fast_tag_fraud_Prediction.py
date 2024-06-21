# Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


# Importing Data
df = pd.read_csv(r'/content/FastagFraudDetection.csv')
df

df.shape

df.info()

df.describe()

# Data types of columns
print("\nData types:")
df.dtypes


# Check for missing values
print("\nMissing values:")
df.isnull().sum()


# Dropping null values from database
df.dropna(inplace=True)


# Checking unique value counts of Vehicle_Type in the database
df['Vehicle_Type'].value_counts()


# Checking unique value counts of Lane_Type in the database
df['Lane_Type'].value_counts()


# Checking unique value counts of Vehicle_Dimensions in the database
df['Vehicle_Dimensions'].value_counts()


# Checking unique value counts of TollBoothID in the database
df['TollBoothID'].value_counts()


# Checking unique value counts of Transaction_Amount in the database
df['Transaction_Amount'].value_counts()


# Checking unique value counts of Fraud_indicator in the database
df['Fraud_indicator'].value_counts()


# Checking unique value counts of Vehicle_Speed in the database
df['Vehicle_Speed'].value_counts()


# Checking unique value counts of Geographical_Location in the database
df['Geographical_Location'].value_counts()


# Plot 1: Distribution of vehicle types
vehicle_type_plot = sns.countplot(x='Vehicle_Type', data=df, ax=ax[0], color=colors[0])
ax[0].set_xlabel('Vehicle Type', fontsize=14)
ax[0].set_ylabel('Frequency', fontsize=14)
ax[0].set_title('Distribution of Vehicle Type', fontsize=16)
ax[0].grid(False)
for container in vehicle_type_plot.containers:
    vehicle_type_plot.bar_label(container)

# Plot 2: Distribution of lane types
lane_type_plot = sns.countplot(x='Lane_Type', data=df, ax=ax[1], color=colors[1])
ax[1].set_xlabel('Lane Type', fontsize=14)
ax[1].set_ylabel('Frequency', fontsize=14)
ax[1].set_title('Distribution of Lane Type', fontsize=16)
ax[1].grid(False)
ax[1].tick_params(axis='x', rotation=45)
for container in lane_type_plot.containers:
    lane_type_plot.bar_label(container)

# Plot 3: Distribution of geographical locations
geo_location_plot = sns.countplot(x='Geographical_Location', data=df, ax=ax[2], color=colors[2])
ax[2].set_xlabel('Geographical Location', fontsize=14)
ax[2].set_ylabel('Frequency', fontsize=14)
ax[2].set_title('Distribution of Geographical Location', fontsize=16)
ax[2].grid(False)
ax[2].tick_params(axis='x', rotation=90)
for container in geo_location_plot.containers:
    geo_location_plot.bar_label(container)

# Plot 4: Distribution of fraud indicators
fraud_indicator_plot = sns.countplot(x='Fraud_indicator', data=df, ax=ax[3], color=colors[3])
ax[3].set_xlabel('Fraud Indicator', fontsize=14)
ax[3].set_ylabel('Frequency', fontsize=14)
ax[3].set_title('Distribution of Fraud Indicator', fontsize=16)
ax[3].grid(False)
for container in fraud_indicator_plot.containers:
    fraud_indicator_plot.bar_label(container)

# Plot 5: Distribution of transaction amounts
transaction_amount_plot = sns.histplot(df['Transaction_Amount'], bins=50, kde=True, ax=ax[4], color=colors[4])
ax[4].set_xlabel('Transaction Amount', fontsize=14)
ax[4].set_ylabel('Frequency', fontsize=14)
ax[4].set_title('Distribution of Transaction Amount', fontsize=16)
ax[4].grid(False)

# Plot 6: Distribution of vehicle speeds
vehicle_speed_plot = sns.histplot(df['Vehicle_Speed'], bins=50, kde=True, ax=ax[5], color=colors[5])
ax[5].set_xlabel('Vehicle Speed', fontsize=14)
ax[5].set_ylabel('Frequency', fontsize=14)
ax[5].set_title('Distribution of Vehicle Speed', fontsize=16)
ax[5].grid(False)

# Plot 7: Distribution of vehicle dimensions
vehicle_dimensions_plot = sns.histplot(df['Vehicle_Dimensions'], bins=50, kde=True, ax=ax[6], color=colors[6])
ax[6].set_xlabel('Vehicle Dimensions', fontsize=14)
ax[6].set_ylabel('Frequency', fontsize=14)
ax[6].set_title('Distribution of Vehicle Dimensions', fontsize=16)
ax[6].grid(False)

# Plot 8: Distribution of amount paid
amount_paid_plot = sns.histplot(df['Amount_paid'], bins=50, kde=True, ax=ax[7], color=colors[7])
ax[7].set_xlabel('Amount Paid', fontsize=14)
ax[7].set_ylabel('Frequency', fontsize=14)
ax[7].set_title('Distribution of Amount Paid', fontsize=16)
ax[7].grid(False)

# Plot 9: Distribution of toll booth IDs
tollbooth_id_plot = sns.countplot(x='TollBoothID', data=df, ax=ax[8], color=colors[8])
ax[8].set_xlabel('Toll Booth ID', fontsize=14)
ax[8].set_ylabel('Frequency', fontsize=14)
ax[8].set_title('Distribution of Toll Booth ID', fontsize=16)
ax[8].grid(False)
ax[8].tick_params(axis='x', rotation=90)
for container in tollbooth_id_plot.containers:
    tollbooth_id_plot.bar_label(container)

# Plot 10: Distribution of hours of transactions
hour_plot = sns.countplot(x='Hour', data=df, ax=ax[9], color=colors[9])
ax[9].set_xlabel('Hour of Transaction', fontsize=14)
ax[9].set_ylabel('Frequency', fontsize=14)
ax[9].set_title('Distribution of Hour of Transaction', fontsize=16)
ax[9].grid(False)
for container in hour_plot.containers:
    hour_plot.bar_label(container)

# Adjust the spacings for a better presentation
plt.tight_layout()

# Show graphs
plt.show()



# Ensure that only numerical columns are considered for the correlation matrix
numerical_df = df.select_dtypes(include=['float64', 'int64'])

# Visualize the correlation matrix
plt.figure(figsize=(12, 10))  # Adjusting the figure size for better readability
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# Create a boxplot to visualize the distribution of Transaction Amount by Fraud Indicator
plt.figure(figsize=(14, 8))  # Set the figure size for better readability
sns.set(style="whitegrid")  # Change the style of the plot to 'whitegrid' for a clean look

# Create the boxplot
box_plot = sns.boxplot(
    x="Fraud_indicator",
    y="Transaction_Amount",
    showmeans=True,
    data=df,
    palette=["#FF6347", "#90EE90"],
    boxprops=dict(edgecolor='k', linewidth=2),  # Customize the box edge color and width
    whiskerprops=dict(color='k', linewidth=2),  # Customize the whisker color and width
    capprops=dict(color='k', linewidth=2),      # Customize the cap color and width
    medianprops=dict(color='yellow', linewidth=2),  # Customize the median line color and width
    meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=10)  # Customize the mean point
)

# Customize the labels and title
plt.xlabel("Fraud Indicator", fontsize=14)  # X-axis label
plt.ylabel("Transaction Amount", fontsize=14)  # Y-axis label
plt.title("Distribution of Transaction Amount by Fraud Indicator", fontsize=16)  # Title of the plot
plt.xticks(rotation=45, fontsize=12)  # Rotate the x-axis labels for better readability
plt.yticks(fontsize=12)  # Customize the y-axis labels font size

# Show the plot
plt.show()


# Set the figure size for better readability
plt.figure(figsize=(14, 8))

# Change the style of the plot to 'whitegrid' for a clean look
sns.set(style="whitegrid")

# Create a regression plot to visualize the relationship between Transaction Amount and Amount Paid
reg_plot = sns.regplot(
    x='Transaction_Amount',
    y='Amount_paid',
    data=df,
    scatter_kws={'s': 50, 'alpha': 0.5, 'color': 'blue'},  # Customize scatter points
    line_kws={'color': 'red', 'lw': 2}  # Customize regression line
)

# Customize the labels and title
plt.xlabel("Transaction Amount", fontsize=14)
plt.ylabel("Amount Paid", fontsize=14)
plt.title("Regression Plot of Transaction Amount vs. Amount Paid", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.show()



# Verify the column names in your DataFrame
print(df.columns)

# Adjust the list of columns to drop accordingly
x = df.drop(['Fraud_indicator', 'Vehicle_Type', 'Lane_Type', 'Geographical_Location','Transaction_Amount', 'Vehicle_Speed', 'Amount_paid', 'Hour'], axis=1)
y = df['Fraud_indicator']
print(x)
print(y)


# Verify the column names in your DataFrame
print(x.columns)

# Convert the 'Day' column (if it exists) to a numerical representation
# If the 'Day' column is not present, adjust the code accordingly
if 'Day' in x.columns:
    x = pd.get_dummies(x, columns=['Day'])

# Select only numerical columns after one-hot encoding (if performed)
x_numerical = x.select_dtypes(include=['float64', 'int64'])

# Now proceed with scaling
scaler = StandardScaler()
scaler.fit(x_numerical)
x_scaled = scaler.transform(x_numerical)



x=x_scaled
y=df['Fraud_indicator']
print(x)
print(y)


#TRAINING THE MODEL
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
print(x.shape,x_test.shape,x_train.shape)


#TRAINING THE MODEL
from sklearn import svm
classifer=svm.SVC(kernel='linear')
#TRAINING THE SVM CLASSIFIER
classifer.fit(x_train,y_train)


#MODEL EVALUATION
#ACCURACY SCORE ON THE TRAINING DATA
x_train_prediction=classifer.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)

print("accuracy score of training data is : ",training_data_accuracy)



#ACCURACY SCORE ON THE TEST DATA
x_test_prediction=classifer.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)

print("accuracy score of test data is : ",test_data_accuracy)



#PREDICTIVE SYSTEM

#METHOD-1
input_data=[[140]]

prediction=classifer.predict(input_data)
print(prediction)



#METHOD-2
input_data=(100)

#changing the input data into numpy array
input_data_asnumpy_arr=np.asarray(input_data)

#reshape the array as we are predicting for 1 instance
input_data_reshaped=input_data_asnumpy_arr.reshape(1,-1)

#standardize the input data
std_data=scaler.transform(input_data_reshaped)

prediction=classifer.predict(std_data)
print(prediction)