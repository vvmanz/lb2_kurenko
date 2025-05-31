import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


random_state = 42
np.random.seed(random_state)
torch.manual_seed(random_state)

features = ['V1real', 'V2real']
df = pd.read_excel("train_augmented.xlsx", decimal=',')
X = df[features].values.astype(np.float32)
y = df['is_type_2'].values.astype(np.int64)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

class SurfaceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, activation):
        super(Net, self).__init__()
        act_fn = nn.ReLU()
        layers = [nn.Linear(input_dim, hidden_dim), act_fn, nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act_fn, nn.Dropout(dropout_rate)]
        layers.append(nn.Linear(hidden_dim, 2))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

hidden_dim = 64
num_layers = 2
dropout_rate = 0.2
activation = 'relu'
batch_size = 32
epochs = 200

optimizer_map = {
    'Adam': lambda params, lr: optim.Adam(params, lr=lr),
    'SGD': lambda params, lr: optim.SGD(params, lr=lr),
    'RMSprop': lambda params, lr: optim.RMSprop(params, lr=lr),
    'Adagrad': lambda params, lr: optim.Adagrad(params, lr=lr)
}

optimizers = list(optimizer_map.keys())
f1_scores = []

for opt_name in optimizers:
    lr = 0.001 if opt_name != 'SGD' else 0.01
    model = Net(input_dim=2, hidden_dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate, activation=activation)
    optimizer = optimizer_map[opt_name](model.parameters(), lr)
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_y)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

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
        f1_scores.append(f1)
        print(f"Optimizer: {opt_name}, F1-score: {f1:.4f}")

plt.figure(figsize=(8, 5))
plt.bar(optimizers, f1_scores, color='lightgreen')
plt.title("F1-score by Optimizer")
plt.xlabel("Optimizer")
plt.ylabel("F1-score")
plt.grid(True, axis='y')
plt.savefig("optimizer_analysis.png")
plt.show()
