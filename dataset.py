import torch
from torch.utils.data import Dataset, DataLoader

class TweetDataset(Dataset):
    def __init__(self, data, encoder):
        self.data = data
        self.encoder = encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        x = self.encoder(item["text"])
        y = item["label"]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )