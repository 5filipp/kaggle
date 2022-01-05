import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melb_houses = pd.read_csv('data/melb_data.csv')
melb_houses.dropna(axis=0)

y = melb_houses.Price
melb_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melb_houses.loc[:, ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]

melbourne_model = DecisionTreeRegressor(random_state=1)
melbourne_model.fit(X, y)

iowa_houses = pd.read_csv('data/iowa_houses.csv')
y = iowa_houses.Price
X = iowa_houses.loc[:, ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']]


print(melbourne_model.predict(X.head()))
