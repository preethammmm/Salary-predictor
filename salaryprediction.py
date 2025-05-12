import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Salary Data.csv')

# Show the first few rows
print(df.head())

#Get basic info
print(df.info())
print(df.describe())

#Handle Missing Values
print(df[df.isnull().any(axis=1)])

#Dropping the rows with missing values
df = df.dropna()

# Choose to drop them by using dropna() or fill the missing values using code below:
# #df['Age'].fillna(df['Age'].median(), inplace=True)
# df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
# df['Education Level'].fillna(df['Education Level'].mode()[0], inplace=True)
# df['Job Title'].fillna(df['Job Title'].mode()[0], inplace=True)
# df['Years of Experience'].fillna(df['Years of Experience'].median(), inplace=True)
# df['Salary'].fillna(df['Salary'].median(), inplace=True)

# Converting the coloumns into numerical format using one-hot encoding
df = pd.get_dummies(df, columns=['Gender', 'Education Level', 'Job Title'], drop_first=True)

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Age', 'Years of Experience']] = scaler.fit_transform(df[['Age', 'Years of Experience']])

#Splitting the data into Features and Target
X = df.drop('Salary', axis=1)
y = df['Salary']

#Train-Test split of the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import joblib
feature_columns = X_train.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')
scaler = StandardScaler()
X_train[['Age', 'Years of Experience']] = scaler.fit_transform(X_train[['Age', 'Years of Experience']])
X_test[['Age', 'Years of Experience']] = scaler.transform(X_test[['Age', 'Years of Experience']])

"""Training a linear regression model"""

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

##Making the predictions 
y_pred = model.predict(X_test)

##Evaluating the model based on the predictions it did and comparing to the actaul data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

##Finding out which features among the dataset are influencing the salary prediction

import matplotlib.pyplot as plt

# For linear regression, coefficients indicate importance
importance = pd.Series(model.coef_, index=X_train.columns)
importance.sort_values(ascending=False).plot(kind='bar', figsize=(12, 6))
plt.title('Feature Importance')
plt.show()

#Exploring other models, A Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred_rf))
print("Random Forest R2:", r2_score(y_test, y_pred_rf))

#Visualizing the original vs predicted slaries by the model to get better view of the perfomance of the model
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

#factors that influence salry predictions, based on the feature importance graph plotted in image(Figure_1.png)

#Obtaaining the names of the features 
feature_names = X_train.columns
importances = rf.feature_importances_

# Identifying the top most important features that afffect the salary
indices = np.argsort(importances)[::-1]
print("Top 10 most important features:")
for i in range(10):
    print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

###Optimizing the Random Forest model using the Randomized SearchCV. 

from sklearn.model_selection import RandomizedSearchCV

# Parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', 0.5]
}

# Using Randomized SearchCV
rf_random = RandomizedSearchCV(RandomForestRegressor(random_state=42), 
                              param_grid, n_iter=20, cv=5, 
                              scoring='neg_mean_absolute_error',
                              random_state=42)
rf_random.fit(X_train, y_train)

# Printing the best parameters adn model
print(f"Best parameters: {rf_random.best_params_}")
best_rf = rf_random.best_estimator_

# Evaluate the model
y_pred_best = best_rf.predict(X_test)
print(f"Tuned RF MAE: {mean_absolute_error(y_test, y_pred_best)}")
print(f"Tuned RF R²: {r2_score(y_test, y_pred_best)}")

"""Implementing more advanced model to help the model perform even better when compared to the previoius models being evaluated"""

#Trying Gradient Boosting as this outperform the previous RandomForest when using the structured data
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb

# Gradient Boosting
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

# XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Compare models
models = {
    'Linear Regression': {'MAE': 11596.52, 'R²': 0.8522},
    'Random Forest': {'MAE': 9745.33, 'R²': 0.8987},
    'Gradient Boosting': {'MAE': mean_absolute_error(y_test, y_pred_gb), 
                          'R²': r2_score(y_test, y_pred_gb)},
    'XGBoost': {'MAE': mean_absolute_error(y_test, y_pred_xgb), 
                'R²': r2_score(y_test, y_pred_xgb)}
}

# Display comparison
for model, metrics in models.items():
    print(f"{model}: MAE = {metrics['MAE']:.2f}, R² = {metrics['R²']:.4f}")


"""Tuning the xgboost's hyperparameters if it can pass the random forest, which is the leading one"""

from sklearn.model_selection import GridSearchCV
import xgboost as xgb

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBRegressor(random_state=42)
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, 
                          scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_xgb = grid_search.best_estimator_
y_pred_xgb = best_xgb.predict(X_test)
print(f"Tuned XGBoost MAE: {mean_absolute_error(y_test, y_pred_xgb)}")
print(f"Tuned XGBoost R²: {r2_score(y_test, y_pred_xgb)}")


###XGBoost Hyperparameter Tuning

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# Define parameter grid
# params = {
#     'n_estimators': randint(100, 500),
#     'max_depth': randint(3, 8),
#     'learning_rate': uniform(0.01, 0.3),
#     'subsample': uniform(0.6, 0.4),
#     'colsample_bytree': uniform(0.6, 0.4),
#     'gamma': uniform(0, 0.5)
# }

# # Initialize model -01
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1)

# # Randomized search
# search = RandomizedSearchCV(
#     xgb_model, 
#     param_distributions=params,
#     n_iter=50,
#     scoring='neg_mean_absolute_error',
#     cv=5,
#     verbose=3
# )
# search.fit(X_train, y_train)

# # Best model
# best_xgb = search.best_estimator_


# # Initialize model -02
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize and tune
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
search = RandomizedSearchCV(xgb_model, param_grid, n_iter=20, cv=5, 
                            scoring='neg_mean_absolute_error', random_state=42)
search.fit(X_train, y_train)

# Best model
best_xgb = search.best_estimator_


##EVALUATINFG THE MODELS

# Predict
y_pred = best_xgb.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Tuned XGBoost MAE: {mae:.2f}")
print(f"Tuned XGBoost R²: {r2:.4f}")

# Compare models
models = {
    'Linear Regression': {'MAE': 11596.52, 'R²': 0.8522},
    'Random Forest': {'MAE': 9745.33, 'R²': 0.8987},
    'XGBoost (Tuned)': {'MAE': mae, 'R²': r2}
}

for model, metrics in models.items():
    print(f"{model}: MAE = {metrics['MAE']:.2f}, R² = {metrics['R²']:.4f}")

# ###Optimizing the model:
# ##-Creating new features to capture the hidden patterns
# # Interaction term: Experience * Education Level
# df['Experience_Ed'] = df['Years of Experience'] * df['Education Level_PhD']
# # Binning Age into categories
# df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60], labels=['20-30', '30-40', '40-50', '50-60'])

# ## -Normalizing the salary distribution, 
# df['Log_Salary'] = np.log(df['Salary'])
# # Update target variable and retrain models

# ###Model interpretation
# #-SHAP analysi
# import shap

# explainer = shap.TreeExplainer(best_rf)  # Use your best Random Forest model
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values, X_test, plot_type='bar')

####SAVING THE MODEL
import joblib
joblib.dump(best_xgb, 'salary_prediction_model.pkl')
model = joblib.load('salary_prediction_model.pkl')
