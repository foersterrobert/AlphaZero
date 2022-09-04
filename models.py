import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, num_resBlocks, game):
        super().__init__()
        self.game = game
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128)
        )
        self.resBlocks = nn.ModuleList([ResBlock(128, 128) for _ in range(num_resBlocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.game.row_count * self.game.column_count, self.game.action_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * self.game.row_count * self.game.column_count, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.startBlock(x)
        for block in self.resBlocks:
            x = block(x)
        p = self.policy_head(x)
        v = self.value_head(x)
        return p, v

    @torch.no_grad()
    def predict(self, state, augment=False):
        encoded_state = self.game.get_encoded_state(state)
        encoded_stateT = torch.tensor(encoded_state).to(self.device)

        if augment:
            augmented_state = self.game.get_augmented_state(state)
            encoded_augmented_state = self.game.get_encoded_state(augmented_state)
            encoded_augmented_stateT = torch.tensor(encoded_augmented_state).to(self.device)
            encoded_stateT = torch.stack((encoded_augmented_stateT, encoded_stateT), dim=0)
        
        policy, value = self(encoded_stateT.reshape(-1, 3, self.game.row_count, self.game.column_count).float())
        if augment:
            policy[0] = torch.flip(policy[0], dims=(0,))
        policy = policy.mean(0)
        value = value.mean(0)
        return torch.softmax(policy, dim=0).data.cpu().numpy(), value.data.cpu().numpy()[0]

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out