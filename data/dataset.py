import torch
from torch.utils.data import Dataset, DataLoader

class VideoSequenceDataset(Dataset):
    def __init__(self, video_sequences, labels):
        self.video_sequences = video_sequences
        self.labels = labels

    def __len__(self):
        return len(self.video_sequences)

    def __getitem__(self, idx):
        return self.video_sequences[idx], self.labels[idx]

def create_data_loaders(train_sequences, train_labels, batch_size=16, train_ratio=0.8):
    """Create training and validation data loaders"""
    dataset = VideoSequenceDataset(train_sequences, train_labels)
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)

    return trainloader, valloader 