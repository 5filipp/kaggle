import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Read the data
data = pd.read_csv('data///melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

model = RandomForestRegressor(random_state=0)
model.fit(X_train, y)
preds = model.predict(X_valid)
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)