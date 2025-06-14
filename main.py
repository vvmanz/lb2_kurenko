import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import optuna
from imblearn.over_sampling import SMOTE
from collections import Counter

epochs = 40
df = pd.read_excel("train_augmented.xlsx", decimal=',')

features = ['I1', 'I2','I3','gx','gy','gz','ax','ay','az', 'V1real', 'V2real','V3real','N1', 'N2', 'N3']
X = df[features].values.astype(np.float32)
y = df['is_type_2'].values.astype(np.int64)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
df.head()


class SurfaceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    elif name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Unknown activation function: {name}")

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, activation_name):
        super(Net, self).__init__()
        activation = get_activation(activation_name)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation)
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, 2))  # логиты
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)


def objective(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 4, 100)
    num_layers = trial.suggest_int('num_layers', 2, 4)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    activation_name = trial.suggest_categorical('activation', ['relu', 'tanh', 'leaky_relu', 'elu'])

    model = Net(
        input_dim=15,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        activation_name=activation_name
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_y)
    weights = torch.tensor(weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    train_loader = DataLoader(SurfaceDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(test_X, dtype=torch.float32)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).numpy()
        f1 = f1_score(test_y, preds)
    return f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=epochs)

print("Best trial:")
print(study.best_trial.params)

best_params = study.best_trial.params
model = Net(
    input_dim=15,
    hidden_dim=best_params['hidden_dim'],
    num_layers=best_params['num_layers'],
    dropout_rate=best_params['dropout_rate'],
    activation_name=best_params['activation']
)

optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_y)
weights = torch.tensor(weights, dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=weights)

train_loader = DataLoader(SurfaceDataset(train_X, train_y), batch_size=best_params['batch_size'], shuffle=True)

val_f1_history = []
model.train()

for epoch in range(epochs):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(test_X, dtype=torch.float32)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).numpy()
        f1 = f1_score(test_y, preds)
        val_f1_history.append(f1)

model.eval()
with torch.no_grad():
    inputs = torch.tensor(test_X, dtype=torch.float32)
    outputs = model(inputs)
    preds = torch.argmax(outputs, dim=1).numpy()

print("Final Accuracy:", accuracy_score(test_y, preds))
print("Final F1 Score:", f1_score(test_y, preds))

torch.save({
    'model_state_dict': model.state_dict(),
    'params': best_params
}, 'model.pt')
print("Модель сохранена в model.pt")

plt.plot(val_f1_history)
plt.xlabel("Epoch")
plt.ylabel("Validation F1 Score")
plt.title("F1 Score per Epoch")
plt.grid(True)
plt.show()
