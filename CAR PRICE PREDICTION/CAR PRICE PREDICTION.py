import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

data = pd.read_csv('CarPrice_Assignment.csv')

data = data.drop('car_ID', axis=1)

data['CarBrand'] = data['CarName'].apply(lambda x: x.split(' ')[0])
data = data.drop('CarName', axis=1)

data['horsepower'] = np.clip(data['horsepower'], 48, 200)
data['compressionratio'] = np.clip(data['compressionratio'], 7, 15)

data['price'] = np.log1p(data['price'])

X = data.drop('price', axis=1)
y = data['price']

num_features = ['symboling', 'wheelbase', 'carlength', 'carwidth', 'carheight', 
                'curbweight', 'enginesize', 'boreratio', 'stroke', 'compressionratio', 
                'horsepower', 'peakrpm', 'citympg', 'highwaympg']
cat_features = ['CarBrand', 'fueltype', 'aspiration', 'doornumber', 'carbody', 
                'drivewheel', 'enginelocation', 'enginetype', 'cylindernumber', 
                'fuelsystem']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

model_lr = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)

print(f"Max predicted log price (Linear Regression): {np.max(y_pred_lr)}")
print(f"Min predicted log price (Linear Regression): {np.min(y_pred_lr)}")

y_pred_lr_capped = np.clip(y_pred_lr, a_min=None, a_max=20)

y_pred_lr_final = np.expm1(y_pred_lr_capped)
y_test_final = np.expm1(y_test)

mse_lr = mean_squared_error(y_test_final, y_pred_lr_final)
r2_lr = r2_score(y_test_final, y_pred_lr_final)

print(f'Linear Regression - Mean Squared Error: {mse_lr:.2f}, R-squared: {r2_lr:.2f}')

model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

print(f"Max predicted log price (Random Forest): {np.max(y_pred_rf)}")
print(f"Min predicted log price (Random Forest): {np.min(y_pred_rf)}")

y_pred_rf_final = np.expm1(y_pred_rf)

mse_rf = mean_squared_error(y_test_final, y_pred_rf_final)
r2_rf = r2_score(y_test_final, y_pred_rf_final)

print(f'Random Forest - Mean Squared Error: {mse_rf:.2f}, R-squared: {r2_rf:.2f}')

model_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

model_xgb.fit(X_train, y_train)

y_pred_xgb = model_xgb.predict(X_test)

print(f"Max predicted log price (XGBoost): {np.max(y_pred_xgb)}")
print(f"Min predicted log price (XGBoost): {np.min(y_pred_xgb)}")

y_pred_xgb_final = np.expm1(y_pred_xgb)

mse_xgb = mean_squared_error(y_test_final, y_pred_xgb_final)
r2_xgb = r2_score(y_test_final, y_pred_xgb_final)

print(f'XGBoost - Mean Squared Error: {mse_xgb:.2f}, R-squared: {r2_xgb:.2f}')
