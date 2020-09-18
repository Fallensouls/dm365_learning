import torchvision
from torch.utils.data import DataLoader

def load_cifar100(num_workers=4, shuffle=True, batch_size=50000):
    train_data = torchvision.datasets.CIFAR100(
        root='./cifar100/',  # 保存或者提取位置
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=True,  # 没下载就下载, 下载了就不用再下了
    )
    test_data = torchvision.datasets.CIFAR100(
        root='./cifar100/',  # 保存或者提取位置
        train=False,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=True,  # 没下载就下载, 下载了就不用再下了
    )

    cifar100_training_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    cifar100_test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=10000)
    return cifar100_training_loader, cifar100_test_loader

def load_cifar10(num_workers=4, shuffle=True, batch_size=50000):
    train_data = torchvision.datasets.CIFAR10(
        root='./cifar100/',  # 保存或者提取位置
        train=True,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=True,  # 没下载就下载, 下载了就不用再下了
    )
    test_data = torchvision.datasets.CIFAR10(
        root='./cifar100/',  # 保存或者提取位置
        train=False,  # this is training data
        transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
        # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
        download=True,  # 没下载就下载, 下载了就不用再下了
    )

    cifar10_training_loader = DataLoader(train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    cifar10_test_loader = DataLoader(test_data, shuffle=shuffle, num_workers=num_workers, batch_size=10000)
    return cifar10_training_loader, cifar10_test_loader