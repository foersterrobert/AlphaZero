from mcts import Node
import torch
import numpy as np

class MCTSParallel:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        
    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])
        valid_moves = self.game.get_valid_moves(states)
        policy *= valid_moves
        policy /= np.sum(policy, axis=1, keepdims=True)
        
        for i, g in enumerate(spGames):
            g.root = Node(self.game, self.args, states[i], visit_count=1)
            g.root.expand(policy[i])
        
        for search in range(self.args['num_mcts_searches']):
            for g in spGames:
                g.node = None
                node = g.root

                while node.is_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
                value = self.game.get_opponent_value(value)
                
                if is_terminal:
                    node.backpropagate(value)
                    
                else:
                    g.node = node
                    
            expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]
                    
            if len(expandable_spGames) > 0:
                states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])
                
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                valid_moves = self.game.get_valid_moves(states)
                policy *= valid_moves
                policy /= np.sum(policy, axis=1, keepdims=True)
                
                value = value.cpu().numpy()
                
            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                node.expand(policy[i])
                node.backpropagate(value[i])
