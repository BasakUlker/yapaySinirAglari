import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(180401058)

class MLP(nn.Module):

    def __init__(self, inputSize):

        super(MLP, self).__init__()
        self.hiddenLayer1 = nn.Linear(inputSize, 100)
        self.relu1 = nn.ReLU()
        self.hiddenLayer2 = nn.Linear(100, 50)
        self.relu2 = nn.ReLU()
        self.outputLayer = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        out = self.hiddenLayer1(x)
        out = self.relu1(out)
        out = self.hiddenLayer2(out)
        out = self.relu2(out)
        out = self.outputLayer(out)
        out = self.sigmoid(out)
        return out

###############################
#buraya kadar soru 5.1
###############################

class CustomDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)
        self.x = self.data.iloc[:, :-1].values
        self.y = self.data.iloc[:, -1].values.reshape(-1, 1)

    def __getitem__(self, idx):
        return torch.Tensor(self.x[idx]), torch.Tensor(self.y[idx])

    def __len__(self):
        return len(self.data)


trainDataset = CustomDataset('/home/basak/İndirilenler/cure_the_princess_train.csv')
trainLoader = DataLoader(trainDataset, batch_size=16, shuffle=True)

testDataset = CustomDataset('/home/basak/İndirilenler/cure_the_princess_test.csv')
testLoader = DataLoader(testDataset, batch_size=16, shuffle=False)

valDataset = CustomDataset('/home/basak/İndirilenler/cure_the_princess_validation.csv')
valLoader = DataLoader(valDataset, batch_size=16, shuffle=False)

model = MLP(inputSize=13)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
minValLoss = 1

for epoch in range(20):

    for i, data in enumerate(trainLoader):

        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    totalValLoss = 0.0
    for i, data in enumerate(valLoader):

        inputs, labels = data
        outputs = model(inputs)
        valLoss = criterion(outputs, labels)
        totalValLoss += valLoss.item()
    
    print('Epoch: %d, Validation Loss: %.3f' % (epoch+1, totalValLoss/len(valLoader)))

total_test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():

    for i, data in enumerate(testLoader):

        inputs, labels = data
        outputs = model(inputs)
        test_loss = criterion(outputs, labels)
        total_test_loss += test_loss.item()
        predicted = torch.round(outputs)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Totals : Test Loss: %.3f, Test Accuracy: %.3f' % (total_test_loss/len(testLoader), correct/total))
