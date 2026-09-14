# ============================================================
# IIT-H Deep Learning Hackathon
# Binary Image Classification
# ------------------------------------------------------------
# Team Members
#   - Khwaja Abdul Samad  (CS25MTECH11014)
#   - Sajid Ali           (CS25MTECH11019)
# ============================================================
 

import os, random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split


# ── Config ──────────────────────────────────────────────────
class CFG:
    img_size   = 96
    batch_size = 64
    epochs     = 50
    lr         = 2e-4
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    seed       = 2026

    milestone_epochs = [35, 40, 45, 50]


# ── Seed ────────────────────────────────────────────────────
def seed_all(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Dataset ─────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ── Model (ResNet + SE attention) ───────────────────────────
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1)


class Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.se   = SEBlock(out_c)
        self.skip = (nn.Conv2d(in_c, out_c, 1, stride, bias=False)
                     if in_c != out_c or stride != 1 else nn.Identity())

    def forward(self, x):
        return torch.relu(self.se(self.conv(x)) + self.skip(x))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = Block(32,  64,  stride=2)
        self.layer2 = Block(64,  128, stride=2)
        self.layer3 = Block(128, 192, stride=2)
        self.layer4 = Block(192, 256, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


# ── Loss ────────────────────────────────────────────────────
class SmoothBCE(nn.Module):
    """Binary cross-entropy with label smoothing."""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        targets = targets.float() * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(1), targets)


# ── Test Dataset ────────────────────────────────────────────
class TestDataset(Dataset):
    """Simple dataset for test images (no labels)."""
    def __init__(self, img_dir, filenames, transform):
        self.img_dir   = img_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.filenames[idx])
        img  = Image.open(path).convert("RGB")
        return self.transform(img)


# ── Main entry point ─────────────────────────────────────────
def generate_predictions(data_dir):

    seed_all(CFG.seed)
    print("Device:", CFG.device)

    MEAN = [0.5, 0.5, 0.5]
    STD  = [0.5, 0.5, 0.5]

    # ── Transforms ──────────────────────────────────────────
    train_tfms = transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomGrayscale(p=0.15),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((CFG.img_size, CFG.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    # ── Load and Split ───────────────────────────────────────
    train_dir = os.path.join(data_dir, "train",  "train")

    paths, labels = [], []
    for label in ["0", "1"]:
        folder = os.path.join(train_dir, label)
        for img in os.listdir(folder):
            paths.append(os.path.join(folder, img))
            labels.append(int(label))

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths,
        labels,
        test_size=0.1,
        stratify=labels,
        random_state=42
    )

    print(f"Train samples : {len(train_paths)}")
    print(f"Val   samples : {len(val_paths)}")
    print(f"Class 0 train : {train_labels.count(0)}   Class 1 train : {train_labels.count(1)}")

    # ── Loaders ─────────────────────────────────────────────
    train_ds = ImageDataset(train_paths, train_labels, train_tfms)
    val_ds   = ImageDataset(val_paths,   val_labels,   val_tfms)

    train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    print(f"Train batches : {len(train_loader)}")
    print(f"Val   batches : {len(val_loader)}")

    # ── Train Setup ─────────────────────────────────────────
    model     = Net().to(CFG.device)
    criterion = SmoothBCE(smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)

    print("Model      :", model.__class__.__name__)
    print("Optimizer  : Adam  lr=", CFG.lr)
    print("Scheduler  : CosineAnnealingLR  T_max=", CFG.epochs, " eta_min=1e-6")
    print("Loss       : SmoothBCE(smoothing=0.05)")

    # ── Training Loop ────────────────────────────────────────
    # Checkpoint saves:
    #   best_acc.pt    — best validation accuracy seen at any epoch
    #   best_loss.pt   — best validation loss seen at any epoch
    #   ckpt_ep35.pt   — snapshot at epoch 35
    #   ckpt_ep40.pt   — snapshot at epoch 40
    #   ckpt_ep45.pt   — snapshot at epoch 45
    #   ckpt_ep50.pt   — snapshot at epoch 50 (= last)
    #
    # These checkpoints are all genuinely different snapshots spread across
    # the training curve, so ensembling them gives diverse, uncorrelated votes.

    best_acc_val  = 0.0
    best_loss_val = float("inf")

    for epoch in range(1, CFG.epochs + 1):

        # ── Train ────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(CFG.device), y.to(CFG.device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        correct   = 0
        total     = 0
        class1_n  = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y  = x.to(CFG.device), y.to(CFG.device)
                out   = model(x)
                loss  = criterion(out, y)
                val_loss += loss.item()
                prob  = torch.sigmoid(out).squeeze(1)
                pred  = (prob > 0.5).long()
                correct  += (pred == y).sum().item()
                total    += y.size(0)
                class1_n += pred.sum().item()

        acc        = correct / total
        class1_r   = class1_n / total
        avg_vloss  = val_loss / len(val_loader)
        cur_lr     = optimizer.param_groups[0]["lr"]

        # ── Scheduler step ───────────────────────────────────
        scheduler.step()

        # ── Save best_acc ────────────────────────────────────
        if acc > best_acc_val:
            best_acc_val = acc
            torch.save(model.state_dict(), "best_acc.pt")

        # ── Save best_loss ───────────────────────────────────
        if avg_vloss < best_loss_val:
            best_loss_val = avg_vloss
            torch.save(model.state_dict(), "best_loss.pt")

        # ── Save milestone checkpoints ───────────────────────
        if epoch in CFG.milestone_epochs:
            ckpt_name = f"ckpt_ep{epoch}.pt"
            torch.save(model.state_dict(), ckpt_name)
            print(f"  => Milestone checkpoint saved: {ckpt_name}")

        print(f"Epoch {epoch:2d}/{CFG.epochs} | "
              f"Acc {acc:.4f} | "
              f"ValLoss {avg_vloss:.4f} | "
              f"Class1 {class1_r:.2f} | "
              f"LR {cur_lr:.2e}")

    print(f"\nTraining complete.")
    print(f"Best val acc  : {best_acc_val:.4f}  → best_acc.pt")
    print(f"Best val loss : {best_loss_val:.4f}  → best_loss.pt")

    # ── Evaluate Each Checkpoint on Validation ───────────────
    def evaluate_checkpoint(model, path, val_loader):
        """Load checkpoint and report val accuracy + class-1 ratio."""
        if not os.path.exists(path):
            print(f"{path} → NOT FOUND (skipping)")
            return
        model.load_state_dict(torch.load(path, map_location=CFG.device))
        model.eval()

        preds, gt = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x   = x.to(CFG.device)
                out = torch.sigmoid(model(x)).cpu().numpy().flatten()
                preds.extend(out)
                gt.extend(y.numpy())

        preds_bin    = [1 if p > 0.5 else 0 for p in preds]
        acc          = np.mean(np.array(preds_bin) == np.array(gt))
        class1_ratio = np.mean(preds_bin)
        print(f"{path:18s}  →  Acc: {acc:.4f}  |  Class1: {class1_ratio:.2f}")

    for ckpt in ["best_acc.pt", "best_loss.pt",
                 "ckpt_ep35.pt", "ckpt_ep40.pt", "ckpt_ep45.pt", "ckpt_ep50.pt"]:
        evaluate_checkpoint(model, ckpt, val_loader)

    # ── TTA Inference (8 variants per checkpoint) ────────────
    test_dir  = os.path.join(data_dir, "test", "test")
    test_imgs = sorted(os.listdir(test_dir))

    s  = CFG.img_size
    sp = int(s * 1.12)   # slightly larger for center-crop TTA

    norm = transforms.Normalize(mean=MEAN, std=STD)

    TTA_TRANSFORMS = [
        # 0: clean
        transforms.Compose([transforms.Resize((s, s)), transforms.ToTensor(), norm]),
        # 1: horizontal flip
        transforms.Compose([transforms.Resize((s, s)), transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(), norm]),
        # 2: vertical flip
        transforms.Compose([transforms.Resize((s, s)), transforms.RandomVerticalFlip(p=1.0),
                            transforms.ToTensor(), norm]),
        # 3: both flips
        transforms.Compose([transforms.Resize((s, s)), transforms.RandomHorizontalFlip(p=1.0),
                            transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), norm]),
        # 4: slight zoom + center crop
        transforms.Compose([transforms.Resize((sp, sp)), transforms.CenterCrop(s),
                            transforms.ToTensor(), norm]),
        # 5: zoom + crop + hflip
        transforms.Compose([transforms.Resize((sp, sp)), transforms.CenterCrop(s),
                            transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
        # 6: rotate 90
        transforms.Compose([transforms.Resize((s, s)), transforms.RandomRotation((90, 90)),
                            transforms.ToTensor(), norm]),
        # 7: rotate 270
        transforms.Compose([transforms.Resize((s, s)), transforms.RandomRotation((270, 270)),
                            transforms.ToTensor(), norm]),
    ]

    print(f"TTA variants : {len(TTA_TRANSFORMS)}")
    print(f"Test images  : {len(test_imgs)}")

    def predict_with_tta(model, ckpt_path):
        """
        Load checkpoint, run all 8 TTA passes, return averaged class-1 probability
        array of shape (N_test,).
        """
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] {ckpt_path} not found")
            return None

        model.load_state_dict(torch.load(ckpt_path, map_location=CFG.device))
        model.eval()

        sum_probs = np.zeros(len(test_imgs), dtype=np.float64)

        for i, tfm in enumerate(TTA_TRANSFORMS):
            ds  = TestDataset(test_dir, test_imgs, tfm)
            ldr = DataLoader(ds, batch_size=CFG.batch_size * 2, shuffle=False,
                             num_workers=4, pin_memory=True)

            batch_probs = []
            with torch.no_grad():
                for x in ldr:
                    x = x.to(CFG.device)
                    prob = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
                    batch_probs.append(prob)

            sum_probs += np.concatenate(batch_probs)

        avg_probs = sum_probs / len(TTA_TRANSFORMS)
        print(f"  {ckpt_path:18s}  →  Class1 ratio: {(avg_probs > 0.5).mean():.2f}")
        return avg_probs

    print("\nRunning TTA inference on all checkpoints...")

    # ── Collect Predictions per Checkpoint ───────────────────
    p_best_acc  = predict_with_tta(model, "best_acc.pt")
    p_best_loss = predict_with_tta(model, "best_loss.pt")
    p_ep35      = predict_with_tta(model, "ckpt_ep35.pt")
    p_ep40      = predict_with_tta(model, "ckpt_ep40.pt")
    p_ep45      = predict_with_tta(model, "ckpt_ep45.pt")
    p_ep50      = predict_with_tta(model, "ckpt_ep50.pt")

    # ── Save Individual & Ensemble Submissions ───────────────
    def save_preds(probs, name):
        """Threshold at 0.5, save CSV, print class distribution."""
        if probs is None:
            print(f"{name}: skipped (checkpoint not found)")
            return
        preds = (probs > 0.5).astype(int)
        ratio = preds.mean()
        print(f"{name:30s}  Class1%: {ratio:.2f}  "
              f"(0:{(preds==0).sum()}  1:{(preds==1).sum()})")
        df = pd.DataFrame({"ID": test_imgs, "Label": preds})
        df.to_csv(name, index=False)

    # Individual checkpoint submissions
    save_preds(p_best_acc,  "sub_best_acc.csv")
    save_preds(p_best_loss, "sub_best_loss.csv")
    save_preds(p_ep35,      "sub_ep35.csv")
    save_preds(p_ep40,      "sub_ep40.csv")
    save_preds(p_ep45,      "sub_ep45.csv")
    save_preds(p_ep50,      "sub_ep50.csv")

    # Ensemble: average all available checkpoint probabilities
    available = [p for p in [p_best_acc, p_best_loss, p_ep35, p_ep40, p_ep45, p_ep50]
                 if p is not None]
    ensemble  = np.mean(available, axis=0)
    save_preds(ensemble, "sub_ensemble.csv")

    print(f"\nEnsemble used {len(available)} checkpoints × 8 TTA = {len(available)*8} total votes per image")


if __name__ == "__main__":
    data_dir = input()
    generate_predictions(data_dir)