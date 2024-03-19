import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d, AvgPool2d, AvgPool3d, MaxPool3d
from torch.utils.data import DataLoader
from torchsummary import summary

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


def get_correct_count(pred, labels):
    _, predicted = torch.max(pred.data, 1)
    return (predicted.cpu() == labels.cpu()).sum().item()


@torch.inference_mode()  # this annotation disable grad computation
def validate(model, test_loader, device="cpu"):
    correct, total = 0, 0
    for imgs, labels in test_loader:
        pred = model(imgs.to(device))
        total += labels.size(0)
        correct += get_correct_count(pred, labels)
    return correct / total


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_stack = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=7, padding=3),
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
    
device = 'cpu'
model = CNN().to(device)


def train_loop(
        dataloader, 
        model, 
        criterion, 
        optimizer, 
        device
):
    num_batches = len(dataloader)

    train_loss = 0
    y_true, y_pred = torch.Tensor(), torch.Tensor()

    for imgs, labels in dataloader:
        # Compute prediction and loss
        pred =  model(imgs.to(device))
        loss =  criterion(pred, labels.to(device))

        # Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # End of your code

        train_loss += loss.item()

        # accumulating true labels to calculate score function
        y_true = torch.cat([y_true, labels], dim=0)

        # getting predicted labels from logits by argmax
        pred_labels = pred.detach().cpu().argmax(dim=1)
        # accumulating predicted labels to calculate score function
        y_pred = torch.cat([y_pred, pred_labels], dim=0)

    train_loss /= num_batches

    return train_loss


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10

for i in range(num_epochs):
  train_loss =  train_loop(
      train_loader,
      model,
      criterion,
      optimizer,
      device
  )
  print(train_loss)


# accuracy = validate(model, test_loader, device)

# print(f"Accuracy on TEST {accuracy:.2f}")

# print(summary(model, (4,350,3)))