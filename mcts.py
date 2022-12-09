import numpy as np
import math

class Node:
    def __init__(self, state, prior, game, args, parent=None, action_taken=None):
        self.state = state
        self.children = []
        self.parent = parent
        self.total_value = 0
        self.visit_count = 0
        self.prior = prior
        self.action_taken = action_taken
        self.game = game
        self.args = args

    def expand(self, action_probs):
        for a, prob in enumerate(action_probs):
            if prob != 0:
                child_state = self.state.copy()
                child_state = self.game.drop_piece(child_state, a, player=1)
                child_state = self.game.get_canonical_state(child_state, -1)
                child = Node(
                    child_state,
                    prob,
                    self.game,
                    self.args,
                    parent=self,
                    action_taken=a,
                )
                self.children.append(child)

    def backpropagate(self, value):
        self.total_value += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(self.game.get_opponent_value(value))

    def is_expandable(self):
        return len(self.children) > 0

    def select_child(self):
        best_score = -np.inf
        best_child = None

        for child in self.children:
            ucb_score = self.get_ucb_score(child)
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child

        return best_child

    def get_ucb_score(self, child):
        prior_score = self.args['c_puct'] * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
        if child.visit_count == 0:
            return prior_score
        return prior_score - (child.total_value / child.visit_count)

class MCTS:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args

    def search(self, state):
        root = Node(state, prior=0, game=self.game, args=self.args)

        action_probs, value = self.model.predict(state, augment=self.args['augment'])
        action_probs = (1 - self.args['dirichlet_epsilon']) * action_probs + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)   
        valid_moves = self.game.get_valid_locations(state)
        action_probs *= valid_moves
        action_probs /= np.sum(action_probs)

        root.expand(action_probs)

        for simulation in range(self.args['num_simulation_games']):
            node = root

            while node.is_expandable():
                node = node.select_child()

            is_terminal, value = self.game.check_terminal_and_value(node.state, node.action_taken)
            value = self.game.get_opponent_value(value) # value was based on enemy winning

            if not is_terminal:
                action_probs, value = self.model.predict(node.state, augment=False)
                valid_moves = self.game.get_valid_locations(node.state)
                action_probs *= valid_moves
                action_probs /= np.sum(action_probs)
                node.expand(action_probs)
            node.backpropagate(value)

        return root
