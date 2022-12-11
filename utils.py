import numpy as np
from kaggle_environments import make, evaluate

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

def test(gameName, players, num_iterations=1):
    if num_iterations == 1:
        env = make(gameName, debug=True)
        env.run(players)
        return env.render(mode="ipython")

    results = np.array(evaluate(gameName, players, num_episodes=num_iterations))[:, 0]
    print(f"""
Player 1 | Wins: {np.sum(results == 1)} | Draws: {np.sum(results == 0)} | Losses: {np.sum(results == -1)}
Player 2 | Wins: {np.sum(results == -1)} | Draws: {np.sum(results == 0)} | Losses: {np.sum(results == 1)}
    """)
