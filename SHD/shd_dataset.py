from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class SHDNpyDataset(Dataset):
    def __init__(self, directory):
        self.files = sorted(Path(directory).glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in {directory}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        x = torch.from_numpy(np.load(path)).float()
        label = int(path.stem.split("_")[-1])
        return x, torch.tensor(label, dtype=torch.long)
