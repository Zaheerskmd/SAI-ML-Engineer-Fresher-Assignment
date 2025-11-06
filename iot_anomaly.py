import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Disable GUI backend

from sklearn.ensemble import IsolationForest
import os

# outputs folder
os.makedirs("outputs", exist_ok=True)

# Synthetic Sensor Data
np.random.seed(42)
time = pd.date_range(start='2025-01-01', periods=1000, freq='H')
sensor1 = np.random.normal(50, 2, 1000)
sensor2 = np.random.normal(60, 3, 1000)

# Inject anomalies (spikes)
anomaly_indices = np.random.choice(1000, 20, replace=False)
sensor1[anomaly_indices] += np.random.normal(20, 5, 20)
sensor2[anomaly_indices] -= np.random.normal(15, 5, 20)

# DataFrame
df = pd.DataFrame({'timestamp': time, 'sensor1': sensor1, 'sensor2': sensor2})

# Features mean and std
df['sensor1_mean'] = df['sensor1'].rolling(window=20).mean()
df['sensor1_std'] = df['sensor1'].rolling(window=20).std()
df['sensor2_mean'] = df['sensor2'].rolling(window=20).mean()
df['sensor2_std'] = df['sensor2'].rolling(window=20).std()
df = df.dropna().reset_index(drop=True)

# Isolation Forest for Anomaly Detection
features = ['sensor1', 'sensor2', 'sensor1_mean', 'sensor1_std', 'sensor2_mean', 'sensor2_std']
model = IsolationForest(contamination=0.02, random_state=42)
df['anomaly'] = model.fit_predict(df[features])

# results
df.to_csv('outputs/anomaly_results.csv', index=False)
print("✅ Results saved to outputs/anomaly_results.csv")

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['sensor1'], label='Sensor 1')
plt.scatter(df['timestamp'][df['anomaly'] == -1], df['sensor1'][df['anomaly'] == -1], color='red', label='Anomalies')
plt.xlabel('Timestamp')
plt.ylabel('Sensor 1 Reading')
plt.title('Anomaly Detection on Synthetic IoT Sensor Data')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/anomaly_plot.png')
print("✅ Plot saved to outputs/anomaly_plot.png")
