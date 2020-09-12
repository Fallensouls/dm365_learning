import torch
from utils import *
from train import *
from PIL import Image
from torchvision import datasets, models, transforms


def main():
    torch.multiprocessing.freeze_support()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = ''
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    num_classes = 309
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.8)

    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.2, patience=3, verbose=True, min_lr=0.0001)

    model_ft = train_model(data_dir, model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=30, device)


if __name__ == "__main__":
    main()
