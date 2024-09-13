import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

data = pd.read_csv('c:/Users/h4laa/OneDrive/Desktop/creditcard.csv')

print(data.isnull().sum())

print(data['Class'].value_counts())

X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


over_sampler = SMOTE(sampling_strategy=0.2)  
under_sampler = RandomUnderSampler(sampling_strategy=0.8)  

steps = [('over', over_sampler), ('under', under_sampler)]
pipeline = Pipeline(steps=steps)

X_resampled, y_resampled = pipeline.fit_resample(X_scaled, y)

print(pd.Series(y_resampled).value_counts())

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)


print("Logistic Regression Performance:")
print(classification_report(y_test, y_pred_logreg, target_names=['Legitimate', 'Fraudulent']))

print("Random Forest Performance:")
print(classification_report(y_test, y_pred_rf, target_names=['Legitimate', 'Fraudulent']))


precision_logreg = precision_score(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

print(f"Logistic Regression -> Precision: {precision_logreg}, Recall: {recall_logreg}, F1-Score: {f1_logreg}")

precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f"Random Forest -> Precision: {precision_rf}, Recall: {recall_rf}, F1-Score: {f1_rf}")


from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

print("Tuned Random Forest Performance:")
print(classification_report(y_test, y_pred_best_rf, target_names=['Legitimate', 'Fraudulent']))

