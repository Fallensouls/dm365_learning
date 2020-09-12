import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.datasets as datasets
import os
from PIL import Image
import matplotlib.pyplot as plt


def get_data_loaders(data_dir=None, batch_size=1):
    transform = {
        'train': transforms.Compose([
            # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),  # Flip the data horizontally
            # TODO if it is needed, add the random crop
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ]),
        'val': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    }
    data, label = get_data_and_label(data_dir)
    image_datasets = {x: FoodDataset(
        transform=transform[x], image_files=data[x], labels=label[x])
        for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes


class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, transform, image_files, labels):  # from torch.utils.data.Dataset
        # 定义好 image 的路径
        self.images = image_files
        self.labels = labels
        self.transform = transform
        return

    def __len__(self):  # from torch.utils.data.Dataset
        return len(self.images)

    def __getitem__(self, idx):  # from torch.utils.data.Dataset
        raw_img = self.images[index]
        img = Image.open(raw_img)
        return self.transform(img), self.labels[index]


# The way to get one batch from the data_loader
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    data_loader = get_data_loader()
    print(iter(data_loader))
    print(len(iter(data_loader)))
#    for i in range(10):
#        batch_x, batch_y = next(iter(data_loader))
#        print(np.shape(batch_x), batch_y)

    i = 0
    for x, y in data_loader:
        if i == 0:
            a = x
            # print(x.tolist()[:1])

        i += 1

    i = 0
    for x, y in data_loader:
        if i == 0:
            b = x
            # print(x.tolist()[:1])
        i += 1

    c = torchvision.transforms.ToPILImage("RGB")(a[0])
    c.show()
    #d = torchvision.transforms.ToPILImage("RGB")(b[0])
    # d.show()
    torch.T
