import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
from models import MRC_CBAM

def main():
    # 加载数据
    print("Loading data...")
    X = np.load("X_data.npy")
    Y = np.load("Y_label.npy")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")

    # 划分数据集: 75% 训练, 15% 验证, 10% 测试 (论文比例)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.25, random_state=42, stratify=Y
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.4, random_state=42, stratify=Y_temp
    )
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # 转换为Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    Y_train = torch.tensor(Y_train, dtype=torch.long).to(DEVICE)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    Y_val = torch.tensor(Y_val, dtype=torch.long).to(DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    Y_test = torch.tensor(Y_test, dtype=torch.long).to(DEVICE)

    # 模型
    model = MRC_CBAM(input_dim=INPUT_LENGTH, num_classes=NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 学习率调度 (论文未明确，但可选)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练循环
    best_val_acc = 0.0
    train_losses, val_accs = [], []

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0
        for i in range(0, len(X_train), BATCH_SIZE):
            xb = X_train[i:i+BATCH_SIZE]
            yb = Y_train[i:i+BATCH_SIZE]
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(X_train)
        train_losses.append(avg_loss)

        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_preds = torch.argmax(val_outputs, dim=1)
            val_acc = accuracy_score(Y_val.cpu(), val_preds.cpu())
        val_accs.append(val_acc)

        # 调整学习率
        scheduler.step(avg_loss)

        print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  -> Best model saved (acc={val_acc:.4f})")

    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_preds = torch.argmax(test_outputs, dim=1).cpu()
    test_acc = accuracy_score(Y_test.cpu(), test_preds)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(Y_test.cpu(), test_preds, target_names=CLASSES))

    # 混淆矩阵
    cm = confusion_matrix(Y_test.cpu(), test_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    # 绘制训练曲线
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    main()