import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('TeePublic_review.csv', encoding='ISO-8859-1')

data['year'] = data['year'].astype(str).str.extract(r'(\d{4})').astype(int)
data['date'] = pd.to_datetime(data[['year', 'month']].assign(DAY=1))

data['clean_review'] = data['review'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

data['clean_review'] = data['clean_review'].fillna('')

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_review'])

y = data['review-label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Sentiment Analysis Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

geo_data = data[['latitude', 'longitude']].dropna()

scaler = StandardScaler()
geo_data_scaled = scaler.fit_transform(geo_data)
pca = PCA(n_components=2)
geo_data_pca = pca.fit_transform(geo_data_scaled)

kmeans = KMeans(n_clusters=5, random_state=42)
geo_labels = kmeans.fit_predict(geo_data_pca)

plt.figure(figsize=(10,6))
plt.scatter(geo_data_pca[:, 0], geo_data_pca[:, 1], c=geo_labels, cmap='viridis', marker='o')
plt.colorbar()
plt.title('Geospatial Clustering of Reviews (K-Means on PCA-reduced data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

data['geo_cluster'] = pd.Series(geo_labels, index=geo_data.index)

monthly_sentiment = data.groupby('date')['review-label'].mean()

plt.figure(figsize=(10,6))
plt.plot(monthly_sentiment.index, monthly_sentiment.values, marker='o', linestyle='-')
plt.title('Average Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Review Rating (1-5)')
plt.grid(True)
plt.show()

cluster_sentiment = data.groupby('geo_cluster')['review-label'].mean()

plt.figure(figsize=(10,6))
sns.barplot(x=cluster_sentiment.index, y=cluster_sentiment.values, palette="Blues_d")
plt.title('Average Sentiment by Geospatial Cluster')
plt.xlabel('Geospatial Cluster')
plt.ylabel('Average Review Rating (1-5)')
plt.show()

data.to_csv('processed_shoppersentiments.csv', index=False)

print("Processed dataset saved to 'processed_shoppersentiments.csv'.")
