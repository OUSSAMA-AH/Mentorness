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
from sklearn.metrics import r2_score



# Importing Data
df = pd.read_csv(r'/content/Salary_Prediction_Data.csv')
df

df.info()

df.describe()

# Checking for null data
df.isnull().sum()

# Dropping null values from database
df.dropna(inplace=True)

# Checking unique value counts of DESIGNATION in the database
df['DESIGNATION'].value_counts()

# Checking unique value counts of UNIT in the database
df['UNIT'].value_counts()

# Checking unique value counts of AGE in the database
df['AGE'].value_counts()

# Checking unique value counts of PAST EXP in the database
df['PAST EXP'].value_counts()

# Checking unique value counts of SEX in the database
df['SEX'].value_counts()

# Checking unique value counts of RATINGS in the database
df['RATINGS'].value_counts()

# Checking unique value counts of LEAVES USED in the database
df['LEAVES USED'].value_counts()

# Checking unique value counts of LEAVES REMAINING in the database
df['LEAVES REMAINING'].value_counts()



# Create a figure with three subgraphs
fig, ax = plt.subplots(3, 1, figsize=(14, 24))

# Color palette
palette = "Set2"

# First subgraph: gender distribution
gender_plot = sns.countplot(x='SEX', data=df, ax=ax[0], palette=palette)
ax[0].set_xlabel('Gender', fontsize=12)
ax[0].set_ylabel('Frequency', fontsize=12)
ax[0].set_title('Distribution of Gender', fontsize=14)
ax[0].grid(False)
# Add the number labels
for container in gender_plot.containers:
    gender_plot.bar_label(container)

# Second subgraph: distribution of designations
designation_plot = sns.countplot(x='DESIGNATION', data=df, ax=ax[1], palette=palette)
ax[1].set_xlabel('Designation', fontsize=12)
ax[1].set_ylabel('Frequency', fontsize=12)
ax[1].set_title('Distribution of Designation', fontsize=14)
ax[1].grid(False)
ax[1].tick_params(axis='x', rotation=45)
# Add the number labels
for container in designation_plot.containers:
    designation_plot.bar_label(container)

# Third subgraph: distribution of units
unit_plot = sns.countplot(x='UNIT', data=df, ax=ax[2], palette=palette)
ax[2].set_xlabel('Unit', fontsize=12)
ax[2].set_ylabel('Frequency', fontsize=12)
ax[2].set_title('Distribution of Unit', fontsize=14)
ax[2].grid(False)
ax[2].tick_params(axis='x', rotation=45)
# Add the number labels
for container in unit_plot.containers:
    unit_plot.bar_label(container)

# Adjust the spacings for a better presentation
plt.tight_layout()

# Show graphs
plt.show()


# Setting the aesthetic style for the plots
sns.set(style="whitegrid")

# Create a figure with four subplots
fig, ax = plt.subplots(3, 1, figsize=(14, 24))

# Salary Distribution
sns.histplot(df['SALARY'], kde=True, bins=30, color='blue', ax=ax[0])
ax[0].set_title('Salary Distribution')
ax[0].set_xlabel('Salary')
ax[0].set_ylabel('Frequency')
ax[0].axvline(df['SALARY'].mean(), color='red', linestyle='--', label=f"Mean: {df['SALARY'].mean():.2f}")
ax[0].axvline(df['SALARY'].median(), color='green', linestyle='-', label=f"Median: {df['SALARY'].median():.2f}")
ax[0].legend()

# Age Distribution
sns.histplot(df['AGE'].dropna(), kde=True, bins=30, color='green', ax=ax[1])
ax[1].set_title('Age Distribution')
ax[1].set_xlabel('Age')
ax[1].set_ylabel('Frequency')
ax[1].axvline(df['AGE'].mean(), color='red', linestyle='--', label=f"Mean: {df['AGE'].mean():.2f}")
ax[1].axvline(df['AGE'].median(), color='blue', linestyle='-', label=f"Median: {df['AGE'].median():.2f}")
ax[1].legend()

# Past Experience Distribution
sns.histplot(df['PAST EXP'], kde=True, bins=30, color='orange', ax=ax[2])
ax[2].set_title('Past Experience Distribution')
ax[2].set_xlabel('Past Experience (years)')
ax[2].set_ylabel('Frequency')
ax[2].axvline(df['PAST EXP'].mean(), color='red', linestyle='--', label=f"Mean: {df['PAST EXP'].mean():.2f}")
ax[2].axvline(df['PAST EXP'].median(), color='blue', linestyle='-', label=f"Median: {df['PAST EXP'].median():.2f}")
ax[2].legend()

# Adjust spacing between subplots
plt.tight_layout()

#show the plot
plt.show()


# Create a figure with three subplots
fig, ax = plt.subplots(3, 1, figsize=(14, 18))

# Create a bar plot for the relationship between Gender and Salary in the first subplot
sns.barplot(x='SEX', y='SALARY', data=df, ax=ax[0], palette=palette, estimator=pd.Series.mean, ci=None)
ax[0].set(title='Relationship between Gender and Salary', xlabel='Gender', ylabel='Mean Salary')
ax[0].tick_params(axis='x', rotation=45)

# Create a box plot for the relationship between Age and Salary in the second subplot
sns.barplot(x='AGE', y='SALARY', data=df, ax=ax[1], color='green')
ax[1].set(title='Relationship between Age and Salary', xlabel='Age', ylabel='Salary')
ax[1].tick_params(axis='x', rotation=45)

# Create a box plot for the relationship between Years of Experience and Salary in the third subplot
sns.barplot(x='PAST EXP', y='SALARY', data=df, ax=ax[2], color='orange')
ax[2].set(title='Relationship between Years of Experience and Salary', xlabel='Years of Experience', ylabel='Salary')
ax[2].tick_params(axis='x', rotation=45)

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()



# Create a figure with three subplots
fig, ax = plt.subplots(3, 1, figsize=(14, 18))

# Create a bar plot for the relationship between Gender and Salary in the first subplot
sns.barplot(x='DESIGNATION', y='SALARY', data=df, ax=ax[0], color='red', estimator=pd.Series.mean, ci=None)
ax[0].set(title='Relationship between Designation and Salary', xlabel='job', ylabel='Mean Salary')
ax[0].tick_params(axis='x', rotation=45)

# Create a box plot for the relationship between Age and Salary in the second subplot
sns.barplot(x='UNIT', y='SALARY', data=df, ax=ax[1], color='blue')
ax[1].set(title='Relationship between Departement and Salary', xlabel='Departement', ylabel='Salary')
ax[1].tick_params(axis='x', rotation=45)

# Create a box plot for the relationship between Years of Experience and Salary in the third subplot
sns.barplot(x='RATINGS', y='SALARY', data=df, ax=ax[2], color='black')
ax[2].set(title='Relationship between Rating and Salary', xlabel='Ratings', ylabel='Salary')
ax[2].tick_params(axis='x', rotation=45)

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()



# Setting the aesthetic style for the plots
sns.set(style="whitegrid")

# Create a larger figure for better clarity
fig, ax = plt.subplots(3, 1, figsize=(20, 30))

# Plot 1: Designation vs Salary vs Unit
sns.barplot(x='DESIGNATION', y='SALARY', data=df, hue='UNIT', ax=ax[0])
ax[0].set_title('Designation vs Salary vs Unit')
ax[0].tick_params(axis='x', rotation=45)

# Plot 2: Designation vs Salary vs Ratings
sns.barplot(x='DESIGNATION', y='SALARY', data=df, hue='RATINGS', ax=ax[1])
ax[1].set_title('Designation vs Salary vs Ratings')
ax[1].tick_params(axis='x', rotation=45)

# Plot 3: Designation vs Salary vs Past Experience
sns.barplot(x='DESIGNATION', y='SALARY', data=df, hue='PAST EXP', ax=ax[2])
ax[2].set_title('Designation vs Salary vs Past Experience')
ax[2].tick_params(axis='x', rotation=45)

# Adjust layout for better fit
plt.tight_layout()

#show the plot
plt.show()



# Setting the aesthetic style for the plots
sns.set(style="whitegrid")

# Create a larger figure for better clarity
fig, ax = plt.subplots(2, 1, figsize=(20, 30))

# Plot 1: Designation vs Salary vs Unit
sns.barplot(x='AGE', y='SALARY', data=df, hue='RATINGS', ax=ax[0])
ax[0].set_title('Designation vs Salary vs Unit')
ax[0].tick_params(axis='x', rotation=45)

# Plot 2: Designation vs Salary vs Ratings
sns.barplot(x='LEAVES USED', y='SALARY', data=df, hue='UNIT', ax=ax[1])
ax[1].set_title('Designation vs Salary vs Ratings')
ax[1].tick_params(axis='x', rotation=45)

# Adjust layout for better fit
plt.tight_layout()

#show the plot
plt.show()



# Encode categorical variables
df_encoded = df.copy()
label_encoder_sex = LabelEncoder()
label_encoder_designation = LabelEncoder()

df_encoded['SEX_ENCODED'] = label_encoder_sex.fit_transform(df['SEX'])
df_encoded['DESIGNATION_ENCODED'] = label_encoder_designation.fit_transform(df['DESIGNATION'])

# Create a figure with four subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Scatter plot for relationship between age and salary
sns.regplot(x='AGE', y='SALARY', data=df_encoded, scatter_kws={'color':'orange'}, line_kws={'color':'green'}, ax=axes[0, 0])
axes[0, 0].set_title('Relationship between Age and Salary')

# Scatter plot for relationship between past experience and salary
sns.regplot(x='PAST EXP', y='SALARY', data=df_encoded, scatter_kws={'color':'red'}, line_kws={'color':'blue'}, ax=axes[0, 1])
axes[0, 1].set_title('Relationship between Past Experience and Salary')

# Scatter plot for relationship between encoded gender and salary
sns.regplot(x='SEX_ENCODED', y='SALARY', data=df_encoded, scatter_kws={'color':'blue'}, line_kws={'color':'red'}, ax=axes[1, 0])
axes[1, 0].set_title('Relationship between Gender (Encoded) and Salary')
axes[1, 0].set_xticks(range(len(label_encoder_sex.classes_)))
axes[1, 0].set_xticklabels(label_encoder_sex.classes_)

# Scatter plot for relationship between encoded designation and salary
sns.regplot(x='DESIGNATION_ENCODED', y='SALARY', data=df_encoded, scatter_kws={'color':'green'}, line_kws={'color':'orange'}, ax=axes[1, 1])
axes[1, 1].set_title('Relationship between Designation (Encoded) and Salary')
axes[1, 1].set_xticks(range(len(label_encoder_designation.classes_)))
axes[1, 1].set_xticklabels(label_encoder_designation.classes_, rotation=45)

plt.tight_layout()
plt.show()



# Detecting the outliers in the salary column using the IQR method
Q1 = df['SALARY'].quantile(0.25)  # First Quartile
Q3 = df['SALARY'].quantile(0.75)  # Third Quartile

# Calculation of Interquartile Range (IQR)
IQR = Q3 - Q1

# Detecting outliers lying 1.5x of IQR above Q3 and below Q1 respectively
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Lower bound: {lower_bound}")
print(f"Upper bound: {upper_bound}")

# Identifying outliers
outliers = df[(df['SALARY'] < lower_bound) | (df['SALARY'] > upper_bound)]
print(f"Number of outliers detected: {len(outliers)}")
print(outliers)

# Optionally, handling the outliers
# For example, we can remove outliers from the dataset
df_cleaned = df[~((df['SALARY'] < lower_bound) | (df['SALARY'] > upper_bound))]

print(f"Number of records after removing outliers: {len(df_cleaned)}")


# Creating dummies for the categorical columns
categorical_columns = ['SEX', 'DESIGNATION', 'UNIT']
dummies = pd.get_dummies(df[categorical_columns], drop_first=True)

# Concatenating the dummies with the original dataframe
data = pd.concat([df, dummies], axis=1)

# Dropping the original categorical columns
data.drop(categorical_columns, inplace=True, axis=1)

# Displaying the first few rows of the prepared dataframe
print(data.head())



# Separating the dataset into features and target

# Dataset conntaining all features from df
features = df.drop('SALARY',axis=1)

# Series containing target variable to be predicted
target = df['SALARY']


# Splitting data into 25% training and 75% test sets
x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.25,random_state=42)
x_train.shape



# Define models and hyperparameters for tuning
model_params = {
    'Linear_Regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'Decision_Tree': {
        'model': DecisionTreeRegressor(),
        'params': {
            'max_depth': [2, 4, 6, 8, 10],
            'random_state': [0, 42],
            'min_samples_split': [2, 5, 10, 20]
        }
    },
    'Random_Forest': {
        'model': RandomForestRegressor(),
        'params': {
            'n_estimators': [10, 20, 50, 80],
            'random_state': [0, 42],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    }
}


# Encode categorical variables
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)  # Handle unknown categories
x_train_encoded = oe.fit_transform(x_train)
x_test_encoded = oe.transform(x_test)

# Create a pipeline with an imputer and the model
imputer = SimpleImputer(strategy='mean')

# Perform grid search for hyperparameters and collect the results
score = []

for model_name, mp in model_params.items():
    # Initialize the model inside the loop
    model = mp['model']
    pipeline = Pipeline([('imputer', imputer), ('model', model)])

    # Correct the hyperparameters
    params = {f'model__{key}': value for key, value in mp['params'].items()}

    clf = GridSearchCV(pipeline, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    clf.fit(x_train_encoded, y_train)

    # Collect the summary of the best parameters and the mean squared error
    score.append({
        'Model': model_name,
        'Params': clf.best_params_,
        'MSE(-ve)': clf.best_score_
    })

    # Evaluate on the test set and print detailed metrics
    y_pred = clf.predict(x_test_encoded)
    print(f"Evaluation metrics for {model_name}:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred)}")
    print(f"MSE: {mean_squared_error(y_test, y_pred)}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False)}")
    print(f"R2 Score: {r2_score(y_test, y_pred)}\n")

# Display the summary DataFrame
print(pd.DataFrame(score))



# Order of the best models
s = pd.DataFrame(score)
sort = s.sort_values(by = 'MSE(-ve)',ascending=False)
sort


# Random Forest model
rfr = RandomForestRegressor(n_estimators=20)
# Impute missing values in x_train_encoded before fitting
x_train_imputed = imputer.fit_transform(x_train_encoded)
rfr.fit(x_train_imputed,y_train) # Use the imputed training data


# Apply the same encoding used on the training data to the test data
x_test_encoded = oe.transform(x_test)

# Impute missing values in the encoded test data
x_test_imputed = imputer.transform(x_test_encoded)

# Now score using the encoded and imputed test data
rfr.score(x_test_imputed, y_test)



# Apply the same encoding used on the training data to the test data
x_test_encoded = oe.transform(x_test)

# Impute missing values in the encoded test data
x_test_imputed = imputer.transform(x_test_encoded)

# Now predict using the encoded and imputed test data
y_pred_rfr = rfr.predict(x_test_imputed)  # Use x_test_imputed instead of x_test

print("Mean Squared Error :", mean_squared_error(y_test, y_pred_rfr))



# Decision Tree model
from sklearn.preprocessing import OneHotEncoder

# Assuming 'x_train' contains categorical features, we'll use OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')  # 'ignore' handles new categories in test data
x_train_encoded = encoder.fit_transform(x_train)

dtr = DecisionTreeRegressor(max_depth=10,min_samples_split=2,random_state=0)
dtr.fit(x_train_encoded,y_train)  # Fit with encoded data


# Apply the same encoding used on the training data to the test data
x_test_encoded = encoder.transform(x_test)  # Use the fitted encoder

# Now score using the encoded test data
dtr.score(x_test_encoded, y_test)  # Use x_test_encoded instead of x_test


# Apply the same encoding used on the training data to the test data
x_test_encoded = encoder.transform(x_test)  # Use the fitted encoder

# Now predict using the encoded test data
y_pred_dtr = dtr.predict(x_test_encoded)  # Use x_test_encoded instead of x_test

print("Mean Squared Error :",mean_squared_error(y_test,y_pred_dtr))
print("Mean Absolute Error :",mean_absolute_error(y_test,y_pred_dtr))
print("Root Mean Squared Error :",mean_squared_error(y_test,y_pred_dtr,squared=False))


# Linear regression model
from sklearn.preprocessing import OneHotEncoder

# Assuming 'x_train' contains categorical features, we'll use OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')  # 'ignore' handles new categories in test data
x_train_encoded = encoder.fit_transform(x_train)

lr = LinearRegression()
lr.fit(x_train_encoded,y_train) # Fit the model with the encoded data


# Apply the same encoding used on the training data to the test data
x_test_encoded = encoder.transform(x_test)  # Use the fitted encoder

# Now score using the encoded test data
lr.score(x_test_encoded, y_test)  # Use x_test_encoded instead of x_test


# Apply the same encoding used on the training data to the test data
x_test_encoded = encoder.transform(x_test)  # Use the fitted encoder

# Now predict using the encoded test data
y_pred_lr = lr.predict(x_test_encoded)  # Use x_test_encoded instead of x_test

print("Mean Squared Error :",mean_squared_error(y_test,y_pred_lr))
print("Mean Absolute Error :",mean_absolute_error(y_test,y_pred_lr))
print("Root Mean Squared Error :",mean_squared_error(y_test,y_pred_lr,squared=False))



# Access the feature importances of Random Forest Regressor
feature_importances = rfr.feature_importances_

# Assuming you have a list of feature names that corresponds to the feature importances
feature_names = list(x_train.columns)

# Sort feature importances in descending order
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = [feature_importances[i] for i in sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]

# Create a bar chart
plt.figure(figsize=(12, 8))
plt.barh(sorted_feature_names[:10], sorted_feature_importances[:10])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importance in Predicting Salary')
plt.gca().invert_yaxis()  # Invert the y-axis for better visualization
plt.show()




