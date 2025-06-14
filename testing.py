import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import StandardScaler

df_test = pd.read_excel("raw/test.xlsx", decimal=',')
to_delete = []
df_test.drop(columns=to_delete, inplace=True, errors='ignore')
df_test['is_type_2'] = (df_test['Type'] == 2).astype(int)
df_test.drop(columns=['Type'], inplace=True)
v1_features = ['I1', 'I2','I3','gx','gy','gz','ax','ay','az', 'V1real', 'V2real','V3real','N1', 'N2', 'N3']
scaler = joblib.load('scaler_v1_features.pkl')
df_test[v1_features] = scaler.transform(df_test[v1_features])
df_test.head()

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
        raise ValueError(f"Unknown activation: {name}")

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate, activation_name='relu'):
        super(Net, self).__init__()
        activation = get_activation(activation_name)
        layers = [nn.Linear(input_dim, hidden_dim), activation, nn.Dropout(dropout_rate)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation, nn.Dropout(dropout_rate)]
        layers += [nn.Linear(hidden_dim, 2)]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

checkpoint = torch.load('model.pt')
params = checkpoint['params']
model = Net(
    input_dim=15,
    hidden_dim=params['hidden_dim'],
    num_layers=params['num_layers'],
    dropout_rate=params['dropout_rate'],
    activation_name=params.get('activation', 'relu')
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

X_test = df_test[v1_features].values.astype(np.float32)
y_true = df_test['is_type_2'].values
X_tensor = torch.tensor(X_test)

with torch.no_grad():
    logits = model(X_tensor)
    y_pred = torch.argmax(logits, dim=1).numpy()
    y_prob = torch.softmax(logits, dim=1)[:, 1].numpy()

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

print(f"\nAccuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", report)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()