import os
import json
import csv
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Model: ResNet-18 for mels
# -----------------------------

try:
    from torchvision.models import resnet18
except ImportError:
    resnet18 = None


class MelResNet18(nn.Module):
    """
    ResNet-18 adapted to 1-channel log-mel spectrograms.
    """
    def __init__(self, num_classes: int = 6):
        super().__init__()
        if resnet18 is None:
            raise RuntimeError(
                "torchvision not available; install torchvision or "
                "switch to a simpler CNN."
            )
        backbone = resnet18(weights=None)
        # change first conv: 3 -> 1 channel
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Linear(in_feats, num_classes)
        self.backbone = backbone

    def forward(self, x):
        # x: [B, 1, 80, T]
        return self.backbone(x)


# -----------------------------
# Helper functions
# -----------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            loss = criterion(out, y)

            total_loss += loss.item() * X.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    cm = confusion_matrix(all_labels, all_preds)
    return total_loss / total, correct / total, all_preds, all_labels, cm


def plot_confusion_matrix(cm, class_names, title, out_path):
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_training_curves(history, title_prefix, out_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Training Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def write_overall_csv(rows, out_path):
    """
    rows: list of dicts with keys: Model, Train on, Test on, Accuracy
    """
    fieldnames = ["Model", "Train on", "Test on", "Accuracy"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_per_class_csv(class_names, prec, rec, f1, support, out_path):
    """
    Write per-class metrics to CSV with headings:
    Class, Precision, Recall, F1, Support
    """
    fieldnames = ["Class", "Precision", "Recall", "F1", "Support"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, name in enumerate(class_names):
            writer.writerow(
                {
                    "Class": name,
                    "Precision": float(prec[i]),
                    "Recall": float(rec[i]),
                    "F1": float(f1[i]),
                    "Support": int(support[i]),
                }
            )


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    real_dir = "real_npy_data"
    synth_dir = "synthetic_npy_data"
    results_dir = "classifier_results_resnet"
    os.makedirs(results_dir, exist_ok=True)

    # ---- Load data ----

    # REAL
    X_real_train = np.load(os.path.join(real_dir, "real_train_features.npy"))
    y_real_train = np.load(os.path.join(real_dir, "real_train_labels.npy"))
    X_real_val = np.load(os.path.join(real_dir, "real_val_features.npy"))
    y_real_val = np.load(os.path.join(real_dir, "real_val_labels.npy"))

    # SYNTHETIC
    X_synth = np.load(os.path.join(synth_dir, "synthetic_data_features.npy"))
    y_synth = np.load(os.path.join(synth_dir, "synthetic_data_labels.npy"))

    # Class names (use real mapping as canonical)
    with open(os.path.join(real_dir, "class_mapping.json"), "r") as f:
        cls_map = json.load(f)
    id_to_name = {v: k for k, v in cls_map.items()}
    class_names = [id_to_name[i] for i in range(len(id_to_name))]

    print("Real train:", X_real_train.shape, "Real val:", X_real_val.shape)
    print("Synthetic:", X_synth.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names)
    num_epochs = 50

    overall_rows = []

    # ==========================================================
    # 1) BASELINE: Train on REAL, Test on REAL (Real→Real)
    # ==========================================================
    print("\n BASELINE: Train on REAL, Test on REAL ")

    real_train_ds = TensorDataset(
        torch.from_numpy(X_real_train).float(),
        torch.from_numpy(y_real_train).long(),
    )
    real_val_ds = TensorDataset(
        torch.from_numpy(X_real_val).float(),
        torch.from_numpy(y_real_val).long(),
    )

    train_loader_real = DataLoader(real_train_ds, batch_size=32, shuffle=True)
    val_loader_real = DataLoader(real_val_ds, batch_size=32, shuffle=False)

    model_base = MelResNet18(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_base = torch.optim.Adam(
        model_base.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler_base = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_base, T_max=40
    )

    best_val_acc_base = 0.0
    best_state_base = None
    hist_base = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(
            model_base, train_loader_real, criterion, optimizer_base, device
        )
        val_loss, val_acc, _, _, _ = evaluate(
            model_base, val_loader_real, criterion, device
        )
        scheduler_base.step()

        hist_base["train_loss"].append(tr_loss)
        hist_base["train_acc"].append(tr_acc)
        hist_base["val_loss"].append(val_loss)
        hist_base["val_acc"].append(val_acc)

        print(
            f"[Baseline] Epoch {epoch+1:03d}/{num_epochs} | "
            f"REAL train loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
            f"REAL val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc_base:
            best_val_acc_base = val_acc
            best_state_base = model_base.state_dict().copy()

    base_model_path = os.path.join(results_dir, "classifier_resnet_baseline_best.pth")
    torch.save(best_state_base, base_model_path)
    print("Saved baseline best model to", base_model_path)
    print("Best baseline REAL val accuracy:", best_val_acc_base)

    model_base.load_state_dict(torch.load(base_model_path, map_location=device))
    base_loss, base_acc, base_preds, base_labels, cm_base = evaluate(
        model_base, val_loader_real, criterion, device
    )

    print(
        f"\n=== BASELINE ResNet – TRAIN on real, TEST on real ===\n"
        f"Real val loss: {base_loss:.4f}, accuracy: {base_acc:.4f}"
    )

    prec_base, rec_base, f1_base, supp_base = precision_recall_fscore_support(
        base_labels, base_preds, labels=list(range(num_classes))
    )
    for i, name in enumerate(class_names):
        print(
            f"[BASE {name}] prec={prec_base[i]:.3f}, "
            f"rec={rec_base[i]:.3f}, f1={f1_base[i]:.3f}, "
            f"support={supp_base[i]}"
        )

    cm_base_path = os.path.join(results_dir, "cm_resnet_baseline_real.png")
    plot_confusion_matrix(
        cm_base,
        class_names,
        "Baseline Confusion Matrix (Train on: Real, Test on: Real)",
        cm_base_path,
    )
    curves_base_path = os.path.join(
        results_dir, "training_curves_resnet_baseline.png"
    )
    save_training_curves(
        hist_base, "Baseline (Real→Real)", curves_base_path
    )

    per_class_base_csv = os.path.join(
        results_dir, "resnet_baseline_per_class.csv"
    )
    write_per_class_csv(
        class_names, prec_base, rec_base, f1_base, supp_base, per_class_base_csv
    )

    overall_rows.append(
        {
            "Model": "ResNet18",
            "Train on": "Real train",
            "Test on": "Real val",
            "Accuracy": float(base_acc),
        }
    )

    # ==========================================================
    # 2) MIXED: Train on REAL + SYNTHETIC, Test on REAL
    # ==========================================================
    print("\n MIXED: Train on REAL + SYNTHETIC, Test on REAL")

    X_mixed = np.concatenate([X_real_train, X_synth], axis=0)
    y_mixed = np.concatenate([y_real_train, y_synth], axis=0)
    print("Mixed train size:", X_mixed.shape[0])

    mixed_train_ds = TensorDataset(
        torch.from_numpy(X_mixed).float(),
        torch.from_numpy(y_mixed).long(),
    )
    mixed_train_loader = DataLoader(mixed_train_ds, batch_size=32, shuffle=True)

    mixed_val_ds = TensorDataset(
        torch.from_numpy(X_real_val).float(),
        torch.from_numpy(y_real_val).long(),
    )
    mixed_val_loader = DataLoader(mixed_val_ds, batch_size=32, shuffle=False)

    model_mixed = MelResNet18(num_classes=num_classes).to(device)
    optimizer_mixed = torch.optim.Adam(
        model_mixed.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler_mixed = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_mixed, T_max=40
    )

    best_val_acc_mixed = 0.0
    best_state_mixed = None
    hist_mixed = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_epoch(
            model_mixed, mixed_train_loader, criterion, optimizer_mixed, device
        )
        val_loss, val_acc, _, _, _ = evaluate(
            model_mixed, mixed_val_loader, criterion, device
        )
        scheduler_mixed.step()

        hist_mixed["train_loss"].append(tr_loss)
        hist_mixed["train_acc"].append(tr_acc)
        hist_mixed["val_loss"].append(val_loss)
        hist_mixed["val_acc"].append(val_acc)

        print(
            f"[MIXED] Epoch {epoch+1:03d}/{num_epochs} | "
            f"train loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
            f"REAL val loss {val_loss:.4f}, acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc_mixed:
            best_val_acc_mixed = val_acc
            best_state_mixed = model_mixed.state_dict().copy()

    mixed_model_path = os.path.join(results_dir, "classifier_resnet_mixed_best.pth")
    torch.save(best_state_mixed, mixed_model_path)
    print("Saved MIXED best model to", mixed_model_path)
    print("Best REAL-val accuracy (MIXED):", best_val_acc_mixed)

    model_mixed.load_state_dict(torch.load(mixed_model_path, map_location=device))
    mixed_loss, mixed_acc, mixed_preds, mixed_labels, cm_mixed = evaluate(
        model_mixed, mixed_val_loader, criterion, device
    )

    print(
        f"\n=== MIXED ResNet – TRAIN on real + synthetic, TEST on real ===\n"
        f"Real val loss: {mixed_loss:.4f}, accuracy: {mixed_acc:.4f}"
    )

    prec_mixed, rec_mixed, f1_mixed, supp_mixed = precision_recall_fscore_support(
        mixed_labels, mixed_preds, labels=list(range(num_classes))
    )
    for i, name in enumerate(class_names):
        print(
            f"[MIXED {name}] prec={prec_mixed[i]:.3f}, "
            f"rec={rec_mixed[i]:.3f}, f1={f1_mixed[i]:.3f}, "
            f"support={supp_mixed[i]}"
        )

    cm_mixed_path = os.path.join(results_dir, "cm_resnet_mixed_real.png")
    plot_confusion_matrix(
        cm_mixed,
        class_names,
        "Confusion Matrix (Train on: Real and Synthetic, Test on: Real)",
        cm_mixed_path,
    )
    curves_mixed_path = os.path.join(
        results_dir, "training_curves_resnet_mixed.png"
    )
    save_training_curves(
        hist_mixed, "MIXED (Real+Synthetic→Real)", curves_mixed_path
    )

    per_class_mixed_csv = os.path.join(
        results_dir, "resnet_mixed_per_class.csv"
    )
    write_per_class_csv(
        class_names, prec_mixed, rec_mixed, f1_mixed, supp_mixed, per_class_mixed_csv
    )

    overall_rows.append(
        {
            "Model": "ResNet18",
            "Train on": "Real train + Synthetic",
            "Test on": "Real val",
            "Accuracy": float(mixed_acc),
        }
    )

    # -------------------------
    # Overall summary CSV
    # -------------------------
    overall_csv_path = os.path.join(results_dir, "resnet_overall_summary.csv")
    write_overall_csv(overall_rows, overall_csv_path)
    print("Saved overall summary CSV to", overall_csv_path)


if __name__ == "__main__":
    main()
