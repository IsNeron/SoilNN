import torch
import torch.nn as nn
import lightning as L
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import DataLoader

from lighting_module import LitBasic
from scripts.prepare_data import prepare_permeability, generate_cfs_array_test, ExperimentalDataset
from pathlib import Path

data_path = Path('D:\Work\KT\Iranian_ml\cfs\cfs_normalized')
labels_path = Path('D:\Work\KT\Iranian_ml\perm\\normalized')
data = generate_cfs_array_test(data_path)
labels = prepare_permeability(labels_path)

data = torch.Tensor(data)
labels = torch.Tensor(labels)


train_dataset = ExperimentalDataset(data[0:3], labels[0:3])
val_dataset = ExperimentalDataset(data[4], labels[4])
test_dataset = ExperimentalDataset(data[5], labels[5])


train_loader = DataLoader(train_dataset, batch_size=1)
val_loader = DataLoader(val_dataset, batch_size=1)
test_loader = DataLoader(test_dataset, batch_size=1)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, padding=3),
            nn.LeakyReLU(),
            MaxPool2d((4,1), stride=(2,1)),
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=5, padding=2),
            nn.LeakyReLU(),
            MaxPool2d((4,1), stride=(2,1)),
            nn.Conv2d(in_channels=12, out_channels=6, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            MaxPool2d((4,1), stride=(2,1)),
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            MaxPool2d((4,1), stride=(2,1)),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            MaxPool2d((10,1), stride=(5,1))
          
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