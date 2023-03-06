import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import log2
from config import args

import numpy as np


class Conv2dDQN(nn.Module):
    def __init__(self):
        super(Conv2dDQN, self).__init__()

        if args.state_representation == 'one-hot':
            self.conv1 = nn.Conv2d(in_channels=int(log2(args.win_tile)) + 1, out_channels=128, kernel_size=2, stride=1)
        else:
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=2, stride=1)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1)

        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, args.n_actions)

        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=2, stride=1)
        # self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=1)
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1)
        #
        # self.fc1 = nn.Linear(512 * 1 * 1, 256)
        # self.fc2 = nn.Linear(256, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        if args.state_representation != 'one-hot':
            state = state.unsqueeze(1)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))

        x = F.relu(self.fc1(x.view(-1, 128 * 2 * 2)))
        actions = self.fc2(x)

        return actions


class LinearDQN(nn.Module):
    def __init__(self):
        super(LinearDQN, self).__init__()

        input_size = np.prod(args.input_dims) * (int(log2(args.win_tile)) + 1) \
            if args.state_representation == 'one-hot' else np.prod(args.input_dims)

        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 64)

        self.fc4 = nn.Linear(64, args.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        self.eval()
        x = self.bn1(F.relu(self.fc1(state)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)

        return actions
