import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from SNN_layers import DSCADense, ReadoutIntegrator
from shd_dataset import SHDNpyDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class SHDDSCASNN(nn.Module):
    def __init__(self, device, branch=4):
        super().__init__()
        self.device = torch.device(device)
        self.dense_1 = DSCADense(700, 64, branch=branch, device=device, vth=1.0)
        self.dense_2 = ReadoutIntegrator(64, 20, device=device)
        nn.init.xavier_normal_(self.dense_2.dense.weight)
        nn.init.constant_(self.dense_2.dense.bias, 0)

    def apply_masks(self):
        self.dense_1.apply_mask()

    def forward(self, x):
        x = x.to(self.device)
        b, seq_length, input_dim = x.shape
        self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        output = torch.zeros(b, 20, device=self.device)
        for t in range(seq_length):
            x_t = x[:, t, :].reshape(b, input_dim)
            _, spike_1 = self.dense_1(x_t)
            mem_out = self.dense_2(spike_1)
            if t > 10:
                output = output + F.softmax(mem_out, dim=1)
        return output

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).view(-1)
            model.apply_masks()
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="./data/SHD")
    p.add_argument("--dt-tag", default="1ms")
    p.add_argument("--branch", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--step-size", type=int, default=20)
    p.add_argument("--lr-gamma", type=float, default=0.5)
    p.add_argument("--save-dir", default="./checkpoints")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    set_seed(a.seed)
    device = torch.device(a.device if a.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    train_loader = DataLoader(
        SHDNpyDataset(Path(a.data_root)/f"train_{a.dt_tag}"),
        batch_size=a.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        SHDNpyDataset(Path(a.data_root)/f"test_{a.dt_tag}"),
        batch_size=a.batch_size, shuffle=False
    )

    model = SHDDSCASNN(device=device, branch=a.branch).to(device)
    criterion = nn.CrossEntropyLoss()
    base_params = [
    model.dense_2.dense.weight,
    model.dense_2.dense.bias,
    model.dense_1.dense.weight,
    model.dense_1.dense.bias,
    ]

    optimizer = torch.optim.Adam([
    {
        'params': base_params,
        'lr': a.lr
    },
    {
        'params': model.dense_2.tau_m,
        'lr': a.lr * 2
    },
    {
        'params': model.dense_1.tau_m,
        'lr': a.lr * 2
    },
    {
        'params': model.dense_1.tau_n,
        'lr': a.lr * 2
    },
    {
        'params': model.dense_1.ahp_gamm,
        'lr': a.lr * 2
    },
    {
        'params': model.dense_1.ahp_kapp,
        'lr': a.lr * 2
    },
    {
        'params': model.dense_1.feedback_factor,
        'lr': a.lr
    },
    ])
    scheduler = StepLR(optimizer, step_size=a.step_size, gamma=a.lr_gamma)

    save_dir = Path(a.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir/"dsca_shd_best.pth"
    best_acc = 0.0

    for epoch in range(a.epochs):
        model.train()
        model.apply_masks()
        total_loss = 0.0
        correct = total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).view(-1)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            model.apply_masks()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()

        scheduler.step()
        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch:03d} | Loss {total_loss/len(train_loader):.4f} | "
            f"Train {train_acc*100:.2f}% | Test {test_acc*100:.2f}% | "
            f"Best {best_acc*100:.2f}%"
        )

    print(f"Best SHD test accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
