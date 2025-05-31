import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_excel("balanced_dataset.xlsx", decimal=',')
df['target'] = (df['Type'] == 2).astype(np.int64)
class_0 = df[df['target'] == 0]
class_1 = df[df['target'] == 1]
class_1_oversampled = class_1.sample(n=len(class_0), replace=True, random_state=42)
balanced_df = pd.concat([class_0, class_1_oversampled], ignore_index=True).sample(frac=1, random_state=42)
print(Counter(balanced_df['target']))
balanced_df.to_excel("binary_balanced_dataset.xlsx", index=False)


# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # =======================
# # 2. Dataset и модель
# # =======================
#
# class SurfaceDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.long)
#
#     def __len__(self):
#         return len(self.y)
#
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
#
#
# class Net(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
#         super(Net, self).__init__()
#         layers = []
#         layers.append(nn.Linear(input_dim, hidden_dim))
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(dropout_rate))
#
#         for _ in range(num_layers - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout_rate))
#
#         layers.append(nn.Linear(hidden_dim, 2))  # 2 выхода: класс 0 и 1
#         self.model = nn.Sequential(*layers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# def objective(trial):
#     hidden_dim = trial.suggest_int('hidden_dim', 4, 64)
#     num_layers = trial.suggest_int('num_layers', 1, 4)
#     lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
#     dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
#     batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
#
#     model = Net(input_dim=2, hidden_dim=hidden_dim, num_layers=num_layers, dropout_rate=dropout_rate)
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#
#     dataset = SurfaceDataset(train_X, train_y)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
#     model.train()
#     for epoch in range(20):
#         for xb, yb in dataloader:
#             optimizer.zero_grad()
#             preds = model(xb)
#             loss = criterion(preds, yb)
#             loss.backward()
#             optimizer.step()
#
#     model.eval()
#     with torch.no_grad():
#         inputs = torch.tensor(test_X, dtype=torch.float32)
#         outputs = model(inputs)
#         preds = torch.argmax(outputs, dim=1).numpy()
#         f1 = f1_score(test_y, preds)
#
#     return f1
#
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)
#
# print("Best trial:")
# print(study.best_trial.params)
#
# # =======================
# # 4. Финальное обучение с лучшими параметрами
# # =======================
#
# best_params = study.best_trial.params
# model = Net(2, best_params['hidden_dim'], best_params['num_layers'], best_params['dropout_rate'])
# optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
# criterion = nn.CrossEntropyLoss()
#
# train_loader = DataLoader(SurfaceDataset(train_X, train_y), batch_size=best_params['batch_size'], shuffle=True)
#
# model.train()
# for epoch in range(50):
#     for xb, yb in train_loader:
#         optimizer.zero_grad()
#         preds = model(xb)
#         loss = criterion(preds, yb)
#         loss.backward()
#         optimizer.step()
#
# model.eval()
# with torch.no_grad():
#     inputs = torch.tensor(test_X, dtype=torch.float32)
#     outputs = model(inputs)
#     preds = torch.argmax(outputs, dim=1).numpy()
#
# print("Final Accuracy:", accuracy_score(test_y, preds))
# print("Final F1 Score:", f1_score(test_y, preds))
