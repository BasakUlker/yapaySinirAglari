import torch
import torch.nn as nn

torch.manual_seed(180401058)
X = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)

class myClass(nn.Module):

    def __init__(self):

        super(myClass, self).__init__()
        self.hidden = nn.Linear(3, 50)
        self.output = nn.Linear(50, 1)

    def forward(self, x):

        x = torch.tanh(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

model = myClass()
output = model(X)

print(output)
