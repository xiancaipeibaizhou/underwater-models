import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

SR = 20000
SEG_LEN = SR
CLASSES = ["ClassA","ClassB","ClassC","ClassD","ClassE"]
DATA_ROOT = r"E:\PyCharm\mipe+mfcc\ShipsEar"

X, Y = [], []
for label, cls in enumerate(CLASSES):
    folder = os.path.join(DATA_ROOT, cls)
    for f in os.listdir(folder):
        if not f.endswith('.wav'): continue
        y, _ = librosa.load(os.path.join(folder,f), sr=SR)
        for start in range(0, len(y), SEG_LEN):
            seg = y[start:start+SEG_LEN]
            if len(seg)<SEG_LEN:
                seg = np.pad(seg, (0, SEG_LEN-len(seg)))
            mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=20)
            feat = mfcc.flatten()[:1000]
            X.append(feat)
            Y.append(label)

X = np.array(X); Y = np.array(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(Y_train, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(1000,128), nn.ReLU(), nn.Linear(128,5))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(X_train)
    print(f"Epoch {epoch}: loss {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    pred = model(torch.tensor(X_test, dtype=torch.float32)).argmax(1)
acc = accuracy_score(Y_test, pred.numpy())
print(f"Test acc: {acc:.4f}")