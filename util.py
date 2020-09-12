# coding:utf-8
import os


def get_data_and_label(path):
    train_path = os.path.join(path, 'train_data.txt')
    val_path = os.path.join(path, 'val_data.txt')

    train_files = []
    val_files = []
    with open(train_path, "r") as f:
        train_files = [line.strip('\n').split('----', 1)
                       for line in f.readlines()]

    with open(val_path, "r") as f:
        val_files = [line.strip('\n').split('----', 1)
                     for line in f.readlines()]
    # 去除最后多余的空行
    train_files.pop()
    val_files.pop()

    # 分成两部分，文件位置ChineseFoodNet/train/000/000000.jpg 和 标签数字 0
    train_data = [x[0] for x in train_files]
    val_data = [x[0] for x in val_files]

    train_label = [x[1] for x in train_files]
    val_label = [x[1] for x in val_files]
    data = {'train': train_data, 'val': val_data}
    label = {'train': train_label, 'val': val_label}
    return data, label


# 测试用
data, label = get_data_and_label('/home/hatsunemiku/dev/dm365/labels')
print(data)
print(label)
