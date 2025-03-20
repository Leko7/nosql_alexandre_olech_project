import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class UCRDataset(Dataset):
    def __init__(self, root_dir, data_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to the root directory containing datasets.
            data_dir (str): Name of the specific dataset.
            split (str): "train" or "test".
            transform (callable, optional): Normalization function.
        """
        self.data_path = f"{os.path.join(os.path.join(root_dir, data_dir), data_dir)}_{split.upper()}.tsv"
        self.transform = transform

        # Read only the first column to get the total number of samples
        with open(self.data_path, "r") as f:
            self.num_samples = sum(1 for _ in f)  # Count lines in file

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Read only the required row
        row = pd.read_csv(self.data_path, sep="\t", header=None, skiprows=idx, nrows=1).values

        # Extract label and features
        y = int(row[0, 0])  # Label (first column)
        x = row[0, 1:].astype(float)  # Features (remaining columns)

        # Apply normalization if provided
        if self.transform:
            x = self.transform(x)

        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y