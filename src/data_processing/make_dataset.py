import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


# Maybe data_processing and loaders variables accessible while functions private

def _load_data():
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    training_set = datasets.MNIST(root='../data/external/raw', train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root='../data/external/raw', train=False, download=True, transform=transform)
    return training_set, test_set


def _create_loaders(batch_size, train_set, test_set):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


training_set, test_set = _load_data()
train_loader, test_loader = _create_loaders(64, training_set, test_set)
