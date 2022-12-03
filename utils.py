import numpy as np

class KaggleAgent:
    def __init__(self, model, game, augment=False, temperature=0):
        self.model = model
        self.game = game
        self.augment = augment
        self.temperature = temperature

    def run(self, obs, conf):
        player = obs['mark'] if obs['mark'] == 1 else -1
        state = np.array(obs['board']).reshape(self.game.row_count, self.game.column_count)
        state[state==2] = -1
        
        state = self.game.get_canonical_state(state, player)
        valid_moves = self.game.get_valid_locations(state)

        policy, value = self.model.predict(state, augment=self.augment)
        policy *= valid_moves
        policy /= np.sum(policy)

        if self.temperature == 0:
            action = int(np.argmax(policy))

        else:
            policy = policy ** (1 / self.temperature)
            policy /= np.sum(policy)
            action = np.random.choice(self.game.action_size, p=policy)

        return action
