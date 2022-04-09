## MSCI 598 - Final Project ##
## Gaurav Mudbhatkal - 20747018 ##

import torch.nn as nn
import torch.nn.functional as F


class StanceDetectionModel(nn.Module):
    def __init__(self, embedding_dim, dropout):
        super(StanceDetectionModel, self).__init__()
        self.layers = nn.Sequential(
            # linear layer with applies transform to output a 100 dimension vector
            nn.Linear(in_features=embedding_dim, out_features=100),
            # dropout layer
            nn.Dropout(p=dropout),
            # applying relu activation
            nn.ReLU(),
            # linear transformation to output 4 features - softmax applied later to choose best class
            nn.Linear(in_features=100, out_features=4),
            # dropout
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        output = self.layers(x)
        return F.log_softmax(output)