import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from SNN_layers import DSCADense, ReadoutIntegrator

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class MNISTDSCASNN(nn.Module):
    def __init__(self, device, branch=4):
        super().__init__()
        self.device = torch.device(device)
        self.dense_1 = DSCADense(1, 64, branch=branch, device=device, vth=1.0)
        self.dense_2 = DSCADense(64, 128, branch=branch, device=device, vth=1.0)
        self.dense_3 = DSCADense(128, 256, branch=branch, device=device, vth=1.0)
        self.dense_4 = ReadoutIntegrator(256, 10, device=device)
        nn.init.xavier_normal_(self.dense_4.dense.weight)
        nn.init.constant_(self.dense_4.dense.bias, 0)

    def apply_masks(self):
        self.dense_1.apply_mask()
        self.dense_2.apply_mask()
        self.dense_3.apply_mask()

    def forward(self, x):
        x = x.to(self.device)
        b, seq_length, input_dim = x.shape
        self.dense_1.set_neuron_state(b)
        self.dense_2.set_neuron_state(b)
        self.dense_3.set_neuron_state(b)
        self.dense_4.set_neuron_state(b)

        output = torch.zeros(b, 10, device=self.device)
        for t in range(seq_length):
            x_t = x[:, t, :].reshape(b, input_dim)
            _, s1 = self.dense_1(x_t)
            _, s2 = self.dense_2(s1)
            _, s3 = self.dense_3(s2)
            mem_out = self.dense_4(s3)
            if t > 3:
                output = output + F.softmax(mem_out, dim=1)
        return output

def build_loaders(data_dir, batch_size):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    test_set = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=tfm)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


def to_sequence(images):
    return images.view(images.size(0), 784, 1)

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = to_sequence(images).to(device)
            labels = labels.to(device)
            model.apply_masks()
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct/total

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data/MNIST")
    p.add_argument("--branch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--step-size", type=int, default=50)
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--save-dir", default="./checkpoints")
    p.add_argument("--device", default=None)
    a = p.parse_args()

    set_seed(a.seed)
    device = torch.device(a.device if a.device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    train_loader, test_loader = build_loaders(a.data_dir, a.batch_size)
    model = MNISTDSCASNN(device=device, branch=a.branch).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    scheduler = StepLR(optimizer, step_size=a.step_size, gamma=a.lr_gamma)

    save_dir = Path(a.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir/"dsca_smnist_best.pth"
    best_acc = 0.0

    for epoch in range(a.epochs):
        model.train()
        model.apply_masks()
        total_loss = 0.0
        correct = total = 0
        for images, labels in train_loader:
            images = to_sequence(images).to(device)
            labels = labels.to(device)
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
        train_acc = correct/total
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)

        print(
            f"Epoch {epoch:03d} | Loss {total_loss/len(train_loader):.4f} | "
            f"Train {train_acc*100:.2f}% | Test {test_acc*100:.2f}% | "
            f"Best {best_acc*100:.2f}%"
        )

    print(f"Best S-MNIST test accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
