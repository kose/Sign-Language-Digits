import torch
from torch import nn
from torch.nn import functional as F

from parameters import dim1, dim2, dim3, dim4, dim5, dropout

class Net(nn.Module):
    def __init__(self, dim1=dim1, dim2=dim2, dim3=dim3, dim4=dim4, dim5=dim5, dropout=dropout):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, dim1, kernel_size=3, padding=1, stride=2) # 64 -> 32
        self.bn = nn.BatchNorm2d(dim1)

        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=3, padding=1, stride=2) # 32 -> 16

        self.conv3 = nn.Conv2d(dim2, dim3, kernel_size=3, padding=1, stride=2) # 16 -> 8

        self.conv4 = nn.Conv2d(dim3, dim4, kernel_size=3, padding=1, stride=2) # 8 -> 4
        self.dropout = nn.Dropout2d(p=dropout)

        self.ln1 = nn.Linear(4 * 4 * dim4, dim5)
        self.ln2 = nn.Linear(dim5, 10)

        # initial wait, bias
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.ln1, self.ln2]:
            nn.init.xavier_normal_(net.weight, gain=1.0) #Xavier(正規分布)
            # nn.init.kaiming_normal_(net.weight) # He(正規分布)
            nn.init.uniform_(net.bias, 0.0, 0.2) # 一様分布


    def forward(self, x):
        batchsize = x.shape[0]
        h = F.relu(self.bn(self.conv1(x)))
        
        h = F.relu(self.conv2(h))
        
        h = F.relu(self.conv3(h))
        
        h = F.relu(self.conv4(h))
        h = self.dropout(h)

        h = h.view(batchsize, -1)
        h = F.relu(self.ln1(h))
        h = self.ln2(h)

        return F.log_softmax(h, dim=1)

    
    def loss_function(self, x, labels):
        h = self.forward(x)
        return F.nll_loss(h, labels)

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
