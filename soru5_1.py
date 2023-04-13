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

