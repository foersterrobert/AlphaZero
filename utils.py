import numpy as np

class KaggleAgent:
    def __init__(self, model, game, augment=False):
        self.model = model
        self.game = game
        self.augment = augment

    def run(self, obs, conf):
        player = obs['mark'] if obs['mark'] == 1 else -1
        state = np.array(obs['board']).reshape(self.game.row_count, self.game.column_count)
        state[state==2] = -1
        
        state = self.game.get_canonical_state(state, player)
        valid_moves = self.game.get_valid_locations(state)

        policy, value = self.model.predict(state, augment=self.augment)
        policy = policy * valid_moves
        action = int(np.argmax(policy))
        return action
