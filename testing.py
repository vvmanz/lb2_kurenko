import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
import joblib

# =======================
# 1. Подгружаем тестовый датасет и обрабатываем
# =======================
df = pd.read_excel("raw/test.xlsx", decimal=',')
features = ['V1real', 'V2real']
df['target'] = (df['Type'] == 2).astype(np.int64)  # 1 — это тип 2, остальные — 0

X_raw = df[features].values.astype(np.float32)
y_true = df['target'].values.astype(np.int64)

# =======================
# 2. Загружаем scaler и применяем
# =======================
scaler = joblib.load("scaler.pkl")
X_scaled = scaler.transform(X_raw)

# =======================
# 3. Воссоздаём модель и загружаем веса
# =======================
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(Net, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)])
        layers.append(nn.Linear(hidden_dim, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Загружаем параметры и веса
checkpoint = torch.load("model.pt")
params = checkpoint['params']
model = Net(input_dim=2,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout_rate=params['dropout_rate'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    inputs = torch.tensor(X_scaled, dtype=torch.float32)
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1).numpy()

# =======================
# 5. Метрики
# =======================
acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds)

print("Accuracy на raw/test.xlsx:", acc)
print("F1 Score на raw/test.xlsx:", f1)
