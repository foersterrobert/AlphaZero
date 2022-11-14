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

    @torch.no_grad()
    def self_play(self, group_size=10):
        self_play_games = [SelfPlayGame(self.game) for _ in range(group_size)]
        self_play_memory = []
        player = 1

        while len(self_play_games) > 0:
            del_list = []

            for game in self_play_games:
                game.root = Node(self.game.get_canonical_state(game.state, player), 1, prior=0, game=self.game, args=self.args)

            for simulation in range(self.args['num_simulation_games']):
                for game in self_play_games:
                    game.encoded_state = None
                    node = game.root

                    while node.is_expandable():
                        node = node.select_child()

                    is_terminal, value = self.game.check_terminal_and_value(node.state, node.action_taken)
                    value = self.game.get_opponent_value(value)

                    if is_terminal:
                        node.backpropagate(value)

                    else:
                        canonical_state = self.game.get_canonical_state(node.state, node.player)
                        game.encoded_state = self.game.get_encoded_state(canonical_state)
                        game.node = node

                self_play_games_predict = [game for game in self_play_games if game.encoded_state is not None]
                if len(self_play_games_predict) > 0:
                    predict_states = [game.encoded_state for game in self_play_games_predict]
                    predict_states = np.stack(predict_states)

                    predict_states = torch.tensor(predict_states, dtype=torch.float32, device=self.device)
                    action_probs, values = self.model(predict_states)
                    action_probs = torch.softmax(action_probs, dim=1)
                    action_probs = action_probs.cpu().numpy()
                    values = values.cpu().numpy()

                for i, game in enumerate(self_play_games_predict):
                    action_probs_game, value_game = action_probs[i], values[i]
                    valid_moves = self.game.get_valid_locations(game.node.state)
                    action_probs_game = action_probs_game * valid_moves
                    action_probs_game = action_probs_game / np.sum(action_probs_game)
                    game.node.expand(action_probs_game)
                    game.node.backpropagate(value_game)

            for game in self_play_games:
                action_probs = [0] * self.game.action_size
                for child in game.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                game.game_memory.append((game.root.state, player, action_probs))

                if self.args['temperature'] == 0:
                    action = np.argmax(action_probs)
                elif self.args['temperature'] == float('inf'):
                    action = np.random.choice([r for r in range(self.game.action_size) if action_probs[r] > 0])
                else:
                    temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                    temperature_action_probs /= np.sum(temperature_action_probs)
                    action = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)

                game.state = self.game.drop_piece(game.state, action, player)

                is_terminal, reward = self.game.check_terminal_and_value(game.state, action)
                if is_terminal:
                    return_memory = []
                    for hist_state, hist_player, hist_action_probs in game.game_memory:
                        return_memory.append((
                            self.game.get_encoded_state(hist_state), hist_action_probs, reward * ((-1) ** (hist_player != player))
                        ))
                        if self.args['augment']:
                            return_memory.append((
                                self.game.get_encoded_state(self.game.get_augmented_state(hist_state)), np.flip(hist_action_probs), reward * ((-1) ** (hist_player != player))
                            ))
                    self_play_memory.extend(return_memory)
                    del_list.append(game)

            for game in del_list:
                self_play_games.remove(game)
                
            player = self.game.get_opponent_player(player)

        return self_play_memory

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

class SelfPlayGame:
    def __init__(self, game):
        self.game_memory = []
        self.state = game.get_initial_state()
        self.root = None
        self.node = None
        self.encoded_state = None