import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d
from scripts.prepare_data import prepare_permeability, generate_cfs_array_test, ExperimentalDataset
from pathlib import Path

data_path = Path('D:\Work\KT\Iranian_ml\cfs\cfs_normalized')
labels_path = Path('D:\Work\KT\Iranian_ml\perm\\normalized')
data = generate_cfs_array_test(data_path)
labels = prepare_permeability(labels_path)

data = torch.Tensor(data)
labels = torch.Tensor(labels)


test = ExperimentalDataset(data[0:1], labels[0:1])
# print(test[0][0].shape)

# leaky_relu = nn.LeakyReLU()

# conv1 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=7, padding=3)
# conv2 = nn.Conv2d(in_channels=8, out_channels=12, kernel_size=5, padding=2)
# conv3 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3, padding=1)
# conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1)
# conv5 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)

# pool1 = MaxPool2d((4,1), stride=(2,1))
# pool2 = MaxPool2d((10,1), stride=(5,1))


# b = conv1(test[0][0])
# print(b.shape)
# b = leaky_relu(b)
# b = pool1(b)
# print(b.shape)
# b = conv2(b)
# print(b.shape)
# b = leaky_relu(b)
# b = pool1(b)
# print(b.shape)
# b = conv3(b)
# print(b.shape)
# b = leaky_relu(b)
# b = pool1(b)
# print(b.shape)
# b = conv4(b)
# print(b.shape)
# b = leaky_relu(b)
# b = pool1(b)
# b = conv5(b)
# print(b.shape)
# b = leaky_relu(b)
# b = pool2(b)
# print(b.shape)


