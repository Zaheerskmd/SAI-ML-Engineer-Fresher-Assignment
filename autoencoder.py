import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)
n = 3000
time = pd.date_range("2023-01-01", periods=n, freq="h")
sensor1 = np.sin(np.linspace(0, 30, n)) + np.random.normal(0, 0.2, n)
sensor2 = np.cos(np.linspace(0, 30, n)) + np.random.normal(0, 0.2, n)
anomaly_indices = np.random.choice(n, 50, replace=False)
sensor1[anomaly_indices] += np.random.normal(3, 1, 50)
sensor2[anomaly_indices] += np.random.normal(3, 1, 50)
data = pd.DataFrame({"timestamp": time, "sensor1": sensor1, "sensor2": sensor2})
data["label"] = 0
data.loc[anomaly_indices, "label"] = 1

# features
data["s1_roll_mean"] = data["sensor1"].rolling(20, min_periods=1).mean()
data["s1_roll_std"] = data["sensor1"].rolling(20, min_periods=1).std()
data["s2_roll_mean"] = data["sensor2"].rolling(20, min_periods=1).mean()
data["s2_roll_std"] = data["sensor2"].rolling(20, min_periods=1).std()
features = ["sensor1","sensor2","s1_roll_mean","s1_roll_std","s2_roll_mean","s2_roll_std"]
X = data[features].fillna(0)
X_scaled = StandardScaler().fit_transform(X)

# Autoencoder
class Autoencoder(nn.Module):
    def __init__(self, in_dim, enc_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, enc_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(enc_dim, in_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder(in_dim=X_scaled.shape[1], enc_dim=3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
X_normal = X_tensor[data["label"] == 0]

for epoch in range(20):
    optimizer.zero_grad()
    recon = model(X_normal)
    loss = criterion(recon, X_normal)
    loss.backward()
    optimizer.step()
    if (epoch+1)%5==0:
        print(f"Epoch {epoch+1}, Loss={loss.item():.5f}")

with torch.no_grad():
    recon = model(X_tensor)
    mse = torch.mean((X_tensor - recon)**2, dim=1).numpy()

data["recon_error"] = mse
thr = np.percentile(mse, 98)
data["AE_anomaly"] = (data["recon_error"] > thr).astype(int)

plt.figure(figsize=(12,6))
plt.plot(data["timestamp"], data["sensor1"], label="Sensor1", color="blue")
plt.scatter(data.loc[data["AE_anomaly"]==1,"timestamp"],
            data.loc[data["AE_anomaly"]==1,"sensor1"],
            color="red", s=20, label="Anomalies")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/pytorch_autoencoder_anomalies.png")
plt.close()

from sklearn.metrics import precision_score, recall_score, f1_score
print("Precision:", precision_score(data["label"], data["AE_anomaly"]))
print("Recall:", recall_score(data["label"], data["AE_anomaly"]))
print("F1:", f1_score(data["label"], data["AE_anomaly"]))
data.to_csv("outputs/pytorch_autoencoder_results.csv", index=False)
print("✅ Results saved in outputs/")
