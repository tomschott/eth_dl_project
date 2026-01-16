from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size, data_dir='data', train_shuffle=True):
    """
    Returns train_loader and test_loader.
    """
    # Define transforms once here
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, transform=transform)
    #train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    #test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle)
    # Using a larger batch size for testing is usually faster and fine for memory
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader