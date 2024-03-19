import torch
import torch.nn as nn
import lightning as L
from torch.nn.modules.pooling import MaxPool2d, AvgPool2d
from torch.utils.data import DataLoader, Dataset

from lighting_module import LitBasic
from scripts.prepare_data import prepare_permeability, generate_cfs_array_test
from pathlib import Path

data_path = Path('D:\Work\KT\Iranian_ml\cfs\cfs_normalized')
labels_path = Path('D:\Work\KT\Iranian_ml\perm\\normalized')
data = generate_cfs_array_test(data_path)
labels = prepare_permeability(labels_path)


#4 350 3 -- 3 3
data = torch.Tensor(data)
labels = torch.Tensor(labels)
# data = data[:, None, :, :, :]
# print(data.shape)
# print(labels.shape)

# conv = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=3)
# relu = nn.LeakyReLU
# pool = MaxPool2d

# b = conv(data)
# c = relu(b)
# d = pool()

# print(b)
# print(b.shape)


class ExperimentalDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)


train_dataset = ExperimentalDataset(data[0:3], labels[0:3])
val_dataset = ExperimentalDataset(data[4], labels[4])
test_dataset = ExperimentalDataset(data[5], labels[5])


# print(next(iter(train_dataset))[0].shape)


train_loader = DataLoader(train_dataset, batch_size=1)
val_loader = DataLoader(val_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = nn.Sequential(
          #nn.Conv2d(in_channels=4, out_channels=10, kernel_size=3),
          AvgPool2d((3,3))
          
      )
        self.nn_layers = nn.ModuleList()

    def forward(self, x):
        scores = self.layers_stack(x)
        return scores
    

L.seed_everything(42)
model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
lit_model = LitBasic(model)
lit_model
trainer = L.Trainer(max_epochs=10, log_every_n_steps=1,)
trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader, )