import torchvision
from torch.utils.data import DataLoader

def load_cifar100(num_workers=4, shuffle=True, batch_size=64):
    train_data = torchvision.datasets.CIFAR100(
        root='./cifar100/',  # 保存或者提取位置
        train=True,  # this is training data
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=True,  # 没下载就下载, 下载了就不用再下了
    )
    test_data = torchvision.datasets.CIFAR100(
        root='./cifar100/',  # 保存或者提取位置
        train=False,  # this is training data
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=True,  # 没下载就下载, 下载了就不用再下了
    )

    cifar100_training_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=50000)
    cifar100_test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=10000)
    return cifar100_training_loader, cifar100_test_loader


def load_cifar10(num_workers=4, shuffle=False, batch_size=50000):
    transform = torchvision.transforms.Compose([
            # torchvision.transforms.RandomResizedCrop(size=120, scale=(0.2, 1.)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    train_data = torchvision.datasets.CIFAR10(
        root='./cifar100/',  # 保存或者提取位置
        train=True,  # this is training data
        transform=transform,
        download=True,  
    )
    test_data = torchvision.datasets.CIFAR10(
        root='./cifar100/',  # 保存或者提取位置
        train=False,  # this is training data
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]),  
        download=True,
    )

    cifar10_training_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=50000)
    cifar10_test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=10000)
    return cifar10_training_loader, cifar10_test_loader


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]