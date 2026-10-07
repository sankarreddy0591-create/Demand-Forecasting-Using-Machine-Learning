
# import requireed lybraies
import pandas as pd
import numpy as np 

# the modeling libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor   

# load the dataset
data = pd.read_csv('Data/demand_forecasting.csv')
print("Data loaded successfully!")
print(data.shape)

# display the first few rows of the dataset
print(data.head())

#display the last few rows of the dataset
print(data.tail())  

#display number of rows and columns in the dataset
print("Number of rows and columns in the dataset:", data.shape)

#dispaly statistical summary of the dataset
print("Statistical summary of the dataset:", data.describe())

#display the data types of each column in the dataset
print("Data types of each column in the dataset:", data.dtypes) 

# check for missing values in the dataset
print("Missing values in the dataset:", data.isnull().sum())    

# check for duplicate rows in the dataset
print("Duplicate rows in the dataset:", data.duplicated().sum())

# check for unique values in each column of the dataset
print("Unique values in each column of the dataset:", data.nunique())

#display the column names of the dataset
print("Column names of the dataset:", data.columns) 

# define feature selection
features=["Price",
          "Discount",
          "Inventory Level",
          "Promotion",
          "Competitor Pricing",
          "Category"
          ]
print("Features selected for modeling:", features)

# define the target variable
target="Demand"
print("Target variable for modeling:", target)  

# define the features "x" and target variable "y"
x = data[features].copy()
y = data[target].copy().squeeze()  # convert to Series if it's a single column

print("Features for modeling:", x)
print("Target variable for modeling:", y)

# check for categorical features in the dataset
# the model label encoder
label_encoders ={}
categorical_cols = x.select_dtypes(include="object").columns
categorical_cols

# 

for col in categorical_cols:
    le=LabelEncoder()
    x[col] = le.fit_transform(x[col])
    label_encoders[col] = le

label_encoders

# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# define the model xgbregressor
xgb = XGBRegressor(objective='reg:squarederror', n_jobs=-1)
xgb

# define the parameters dictionary 
param_dict = {
    'n_estimators': [200, 300,500],
    'max_depth': [3, 4, 5,8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
}

param_dict


# the model random search cv
random_search = RandomizedSearchCV(
    xgb, 
    param_distributions=param_dict,
    n_iter=25, 
    scoring='neg_mean_squared_error',
    cv=3, 
    verbose=1, 
    n_jobs=-1
)

random_search

# fit the model to the training data
random_search.fit(x_train, y_train)

random_search.best_params_

best_model = random_search.best_estimator_
best_model

# make predictions on the test set
y_pred = best_model.predict(x_test)
y_pred

# evaluate the model performance
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)   
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

