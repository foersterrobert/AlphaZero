import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import trange
from mcts import MCTS

class Trainer:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.mcts = MCTS(self.model, self.game, self.args)

    def self_play(self):
        game_memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            root = self.mcts.search(self.game.get_canonical_state(state, player), 1)
            action_probs = [0] * self.game.action_size
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)

            game_memory.append((root.state, player, action_probs))
            if self.args['augment']:
                game_memory.append((self.game.get_augmented_state(root.state), player, np.flip(action_probs)))

            visit_counts = [child.visit_count for child in root.children]
            actions = [child.action_taken for child in root.children]
            if self.args['temperature'] == 0:
                action = actions[np.argmax(visit_counts)]
            elif self.args['temperature'] == float('inf'):
                action = np.random.choice(actions)
            else:
                visit_count_distribution = np.array(visit_counts) ** (1 / self.args['temperature'])
                visit_count_distribution = visit_count_distribution / sum(visit_count_distribution)
                action = np.random.choice(actions, p=visit_count_distribution)

            state = self.game.drop_piece(state, action, player)

            is_terminal, reward = self.game.check_terminal_and_value(state, action)
            if is_terminal:
                return_memory = []
                for hist_state, hist_player, hist_action_probs in game_memory:
                    return_memory.append((
                        self.game.get_encoded_state(hist_state), hist_action_probs, reward * ((-1) ** (hist_player != player))
                    ))
                return return_memory

            player = self.game.get_opponent_player(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory) -1, self.args['batch_size']):
            state, policy, value = list(zip(*memory[batchIdx:min(len(memory) -1, batchIdx + self.args['batch_size'])]))
            state, policy, value = np.array(state), np.array(policy), np.array(value).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            policy = torch.tensor(policy, dtype=torch.float32).to(self.device)
            value = torch.tensor(value, dtype=torch.float32).to(self.device)

            out_policy, out_value = self.model(state)
            loss_policy = F.cross_entropy(out_policy, policy) 
            loss_value = F.mse_loss(out_value, value)
            loss = loss_policy + loss_value

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run(self):
        for iteration in range(self.args['num_iterations']):
            print(f"iteration: {iteration}")
            memory = []

            self.model.eval()
            for train_game in trange(self.args['num_train_games'], desc="train_game"):
                memory += self.self_play()

            self.model.train()
            for epoch in trange(self.args['num_epochs'], desc="epoch"):
                self.train(memory)

            torch.save(self.model.state_dict(), f"Models/{self.game}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/{self.game}/optimizer_{iteration}.pt")
