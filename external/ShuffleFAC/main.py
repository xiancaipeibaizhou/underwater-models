import os, csv
import torch

from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
import yaml
from tqdm import tqdm
from codecarbon import OfflineEmissionsTracker

from model.shuffleFAC import shuffleFAC 
from utils.data_preprocessing import dataset
from utils.utils import calculate_macs, count_parameters


def train(student, train_loader, optimizer, criterion, device):
    student.train()
    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader), desc='Train', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        optimizer.zero_grad()

        student_outputs = student(batch_x)
        loss_student = criterion(student_outputs, batch_y)

        loss_student.backward()
        optimizer.step()
        total_loss += loss_student.item()
        num_batches += 1
        
    return total_loss / max(1, num_batches)

@torch.no_grad()
def evaluate(student, val_loader, criterion, device):
    student.eval()
    val_loss = 0.0
    num_batches = 0
    y_true_all = []
    y_pred_all = []

    for batch_x, batch_y in tqdm(val_loader, total=len(val_loader), desc='Valid', leave=False, dynamic_ncols=True):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        outputs = student(batch_x)
        loss = criterion(outputs, batch_y)
        val_loss += loss.item()
        num_batches += 1
        pred = torch.argmax(outputs, dim=1)
        y_true_all.append(batch_y.detach().cpu())
        y_pred_all.append(pred.detach().cpu())

    if len(y_true_all) == 0:
        return 0.0, 0.0, 0.0

    y_true = torch.cat(y_true_all, dim=0).numpy()
    y_pred = torch.cat(y_pred_all, dim=0).numpy()
    val_acc = accuracy_score(y_true, y_pred)
    val_macro_f1 = f1_score(y_true, y_pred, average='macro')

    return (val_loss / max(1, num_batches)), val_acc, val_macro_f1

def get_next_exp_dir(base_dir: str = "exp_save_path") -> str:
    os.makedirs(base_dir, exist_ok=True)
    version = 1
    while True:
        exp_dir_path = os.path.join(base_dir, str(version))
        if not os.path.exists(exp_dir_path):
            os.makedirs(exp_dir_path)
            return exp_dir_path
        version += 1

def main():

    with open('yaml_path', 'r') as f:
        configs = yaml.safe_load(f)
    cnn_cfg = configs["CNN"]
    feats_cfg = configs["feats"]

    DATA_ROOT = "data_path"
    train_set = dataset(os.path.join(DATA_ROOT, "train"), mel_kwargs=feats_cfg)
    val_set = dataset(os.path.join(DATA_ROOT, "val"), mel_kwargs=feats_cfg)
    test_set = dataset(os.path.join(DATA_ROOT, "test"), mel_kwargs=feats_cfg)

    print(f"train/val/test sizes: {len(train_set)}/{len(val_set)}/{len(test_set)}", flush=True)
    if len(train_set) == 0:
        print("[WARN] Dataset is empty. Check path "
              f"'{DATA_ROOT}' and class folder names.", flush=True)

    # Dataloaders
    num_workers = 4
    train_loader = DataLoader(train_set, batch_size=48, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=48, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=48, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    student = shuffleFAC(**cnn_cfg)
    print(student)
    macs, _ = calculate_macs(student, configs)
    total_params, trainable_params = count_parameters(student)

    print("---------------------------------------------------------------")
    print("Model Information:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"MACs: {macs}")
    print("---------------------------------------------------------------\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    student = student.to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Logging / checkpoints
    exp_dir = get_next_exp_dir()
    log_path = os.path.join(exp_dir, 'training_log.csv')
    best_ckpt_path = os.path.join(exp_dir, 'best.pt') # 체크포인트 이름은 best.pt로 고정
    print(f"이번 실험 결과는 여기에 저장됩니다: {exp_dir}")

    best_f1 = -1.0
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc', 'val_macro_f1'])

    # Emissions tracker (covers train+val+test)
    tracker = OfflineEmissionsTracker(
        "Deepship",
        output_dir=exp_dir, # 현재 실험 폴더에 저장
        log_level="warning",
        country_iso_code="KOR",
    )
    tracker.start()

    num_epochs = 200
    for epoch in range(num_epochs):
        print(f"[Epoch {epoch+1}/{num_epochs}] start", flush=True)
        train_loss = train(student,train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(student, val_loader, criterion, device)

        print(f"epoch {epoch+1}: "
              f"train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} "
              f"val_f1={val_f1:.4f}")

        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}"])

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {'epoch': epoch + 1, 'model_state': student.state_dict(), 'best_f1': best_f1},
                best_ckpt_path # 고정된 경로에 덮어쓰기
            )
    # Load best checkpoint for testing
    best_ckpt = os.path.join(best_ckpt_path, 'best.pt')
    if os.path.exists(best_ckpt):
        state = torch.load(best_ckpt, map_location=device)
        student.load_state_dict(state['model_state'])
        print(f"Loaded best checkpoint (epoch={state.get('epoch')}, best_f1={state.get('best_f1'):.4f})")

    test_loss, test_acc, test_macro_f1 = evaluate(student, test_loader, criterion, device)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_macro_f1:.4f}")

    emissions = tracker.stop()
    print(f"[CodeCarbon] Estimated emissions: {emissions} kg CO2eq")

if __name__ == "__main__":
    main()
