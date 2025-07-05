import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
df = pd.read_csv("income.csv")

# Step 2: Select relevant features
X = df[['Age', 'Income($)']]  # Using only numeric columns

# Step 3: Feature Scaling (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 5: Visualize the clusters
plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['Income($)'], c=df['Cluster'], cmap='viridis', s=100, edgecolor='k')
plt.title('K-Means Clustering on Income Dataset')
plt.xlabel('Age')
plt.ylabel('Income ($)')
plt.grid(True)
plt.colorbar(label='Cluster')
plt.show()

# Step 6: Display clustered data
print(df[['Name', 'Age', 'Income($)', 'Cluster']])
