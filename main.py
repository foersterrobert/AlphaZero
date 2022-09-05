import torch
from torch.optim import Adam
import random
import numpy as np
from games import ConnectFour, TicTacToe
from models import ResNet
from trainer import Trainer

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GAME = 'ConnectFour'
LOAD = False

if __name__ == '__main__':
    if GAME == 'ConnectFour':
        args = {
            'num_iterations': 48,             # number of highest level iterations
            'num_train_games': 500,           # number of self-play games to play within each iteration
            'num_simulation_games': 600,      # number of mcts simulations when selecting a move within self-play
            'num_epochs': 4,                  # number of epochs for training on self-play data for each iteration
            'batch_size': 128,                # batch size for training
            'temperature': 1,                 # temperature for the softmax selection of moves
            'c_puct': 2,                      # the value of the constant policy
            'augment': False,                 # whether to augment the training data with flipped states
        }

        game = ConnectFour()
        model = ResNet(9, game).to(device)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        if LOAD:
            model.load_state_dict(torch.load(f'Models/{game}/model.pt', map_location=device))
            optimizer.load_state_dict(torch.load(f'Models/{game}/optimizer.pt', map_location=device))

    elif GAME == 'TicTacToe':
            args = {
                'num_iterations': 8,              # number of highest level iterations
                'num_train_games': 500,           # number of self-play games to play within each iteration
                'num_simulation_games': 60,       # number of mcts simulations when selecting a move within self-play
                'num_epochs': 4,                  # number of epochs for training on self-play data for each iteration
                'batch_size': 64,                 # batch size for training
                'temperature': 1,                 # temperature for the softmax selection of moves
                'c_puct': 2,                      # the value of the constant policy
                'augment': False,                 # whether to augment the training data with flipped states
            }

            game = TicTacToe()
            model = ResNet(4, game).to(device)
            optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            if LOAD:
                model.load_state_dict(torch.load(f'Models/{game}/model.pt', map_location=device))
                optimizer.load_state_dict(torch.load(f'Models/{game}/optimizer.pt', map_location=device))

    trainer = Trainer(model, optimizer, game, args)
    trainer.run()
