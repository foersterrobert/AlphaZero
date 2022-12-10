import numpy as np
import random
import torch
import torch.nn.functional as F
from tqdm import trange
from mcts import Node

class Trainer:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.memory = []

    @torch.no_grad()
    def self_play(self):
        self_play_games = [SelfPlayGame(self.game) for _ in range(self.args['group_size'])]
        player = 1

        while len(self_play_games) > 0:
            states = np.stack([self_play_game.state for self_play_game in self_play_games])
            canonical_states = self.game.get_canonical_state(states, player)
            encoded_states = self.game.get_encoded_state(canonical_states)

            encoded_states = torch.tensor(encoded_states, dtype=torch.float32, device=self.device)
            action_probs, value = self.model(encoded_states)
            action_probs = torch.softmax(action_probs, dim=1).cpu().numpy()

            for i, self_play_game in enumerate(self_play_games):
                self_play_game.root = Node(
                    canonical_states[i],
                    prior=0, game=self.game, args=self.args
                )

                my_action_probs = action_probs[i]
                my_action_probs = (1 - self.args['dirichlet_epsilon']) * my_action_probs + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
                my_valid_locations = self.game.get_valid_locations(self_play_game.root.state)
                my_action_probs *= my_valid_locations
                my_action_probs /= np.sum(my_action_probs)

                self_play_game.root.expand(my_action_probs)

            for simulation in range(self.args['num_simulation_games']):
                for self_play_game in self_play_games:
                    node = self_play_game.root

                    while node.is_expandable():
                        node = node.select_child()

                    self_play_game.is_terminal, value = self.game.check_terminal_and_value(node.state, node.action_taken)
                    value = self.game.get_opponent_value(value)

                    if self_play_game.is_terminal:
                        node.backpropagate(value)

                    else:
                        self_play_game.node = node

                selected_self_play_games = [
                    i for i in range(len(self_play_games)) if self_play_games[i].is_terminal == False 
                ]
                
                if len(selected_self_play_games) > 0:
                    states = np.stack([self_play_games[i].node.state for i in selected_self_play_games])
                    encoded_states = self.game.get_encoded_state(states)
                    
                    encoded_states = torch.tensor(encoded_states, dtype=torch.float32, device=self.device)
                    action_probs, value = self.model(encoded_states)
                    action_probs = torch.softmax(action_probs, dim=1).cpu().numpy()
                    value = value.cpu().numpy()

                for idx, i in enumerate(selected_self_play_games):
                    my_action_probs, my_value = action_probs[idx], value[idx]
                    valid_moves = self.game.get_valid_locations(self_play_games[i].node.state)
                    my_action_probs *= valid_moves
                    my_action_probs /= np.sum(my_action_probs)

                    self_play_games[i].node.expand(my_action_probs)
                    self_play_games[i].node.backpropagate(my_value)

            for i in range(len(self_play_games) -1, -1, -1):
                self_play_game = self_play_games[i]
                action_probs = [0] * self.game.action_size
                for child in self_play_game.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                self_play_game.memory.append((self_play_game.root.state, player, action_probs))

                if self.args['temperature'] == 0:
                    action = np.argmax(action_probs)
                elif self.args['temperature'] == float('inf'):
                    action = np.random.choice([r for r in range(self.game.action_size) if action_probs[r] > 0])
                else:
                    temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                    temperature_action_probs /= np.sum(temperature_action_probs)
                    action = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)

                self_play_game.state = self.game.drop_piece(self_play_game.state, action, player)

                is_terminal, reward = self.game.check_terminal_and_value(self_play_game.state, action)
                if is_terminal:
                    for hist_state, hist_player, hist_action_probs in self_play_game.memory:
                        self.memory.append((
                            self.game.get_encoded_state(hist_state), hist_action_probs, reward * ((-1) ** (hist_player != player))
                        ))
                        if self.args['augment']:
                            self.memory.append((
                                self.game.get_encoded_state(self.game.get_augmented_state(hist_state)), np.flip(hist_action_probs), reward * ((-1) ** (hist_player != player))
                            ))
                    del self_play_games[i]

            player = self.game.get_opponent_player(player)

    def train(self):
        random.shuffle(self.memory)
        for batchIdx in range(0, len(self.memory) -1, self.args['batch_size']):
            state, policy, value = list(zip(*self.memory[batchIdx:min(len(self.memory) -1, batchIdx + self.args['batch_size'])]))
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
            self.memory = []

            self.model.eval()
            for train_game in trange(self.args['num_train_games'] // self.args['group_size'], desc="train_game"):
                self.self_play()

            self.model.train()
            for epoch in trange(self.args['num_epochs'], desc="epoch"):
                self.train()

            torch.save(self.model.state_dict(), f"Models/{self.game}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"Models/{self.game}/optimizer_{iteration}.pt")

class SelfPlayGame:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None