import numpy as np

class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4

    def __repr__(self):
        return 'ConnectFour'

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=np.int8)

    def is_position_a_winner(self, state, action):
        if action is None:
            return False
            
        # row = np.argmax(state[:, action] != 0)
        row = min([r for r in range(self.row_count) if state[r][action] != 0])
        mark = state[row][action]
            
        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row_count
                    or c < 0 
                    or c >= self.column_count
                    or state[r][c] != mark
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1 # vertical
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1 # horizontal
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1 # top left diagonal
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1 # top right diagonal
        )
    
    def drop_piece(self, state, action, player):
        # row = np.argmax(state[:, 0] != 0) - 1
        row = max([r for r in range(self.row_count) if state[r][action] == 0])
        state[row][action] = player
        return state

    def get_valid_locations(self, state):
        return (state[0] == 0).astype(np.uint8)

    def get_canonical_state(self, state, player):
        return state * player

    def get_encoded_state(self, observation):
        if len(observation.shape) == 3:
            encoded_observation = np.swapaxes(np.stack(
                ((observation == -1), (observation == 0), (observation == 1))), 0, 1
            ).astype(np.float32)

        else:
            encoded_observation = np.stack((
                (observation == -1),
                (observation == 0),
                (observation == 1)
            )).astype(np.float32)

        return encoded_observation

    def get_augmented_state(self, state):
        return np.flip(state, axis=1)

    def get_opponent_value(self, score):
        return -1*score

    def get_opponent_player(self, player):
        return -1*player

    def check_terminal_and_value(self, state, action):
        if self.is_position_a_winner(state, action):
            return (True, 1)
        if np.sum(self.get_valid_locations(state)) == 0:
            return (True, 0)
        return (False, 0)
    
class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = 9

    def __repr__(self):
        return 'TicTacToe'

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count), dtype=np.int8)

    def is_position_a_winner(self, state, action):
        if action is None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        mark = state[row][column]
        
        return (
            np.sum(state[row]) == mark * self.column_count # row
            or np.sum(state[:, column]) == mark * self.row_count # column 
            or np.sum(np.diag(state)) == mark * self.row_count # diagonal 
            or np.sum(np.diag(np.fliplr(state))) == mark * self.row_count # flipped diagonal
        )

    def drop_piece(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row][column] = player
        return state

    def get_valid_locations(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def get_canonical_state(self, state, player):
        return state * player

    def get_encoded_state(self, observation):
        if len(observation.shape) == 3:
            encoded_observation = np.swapaxes(np.stack(
                ((observation == -1), (observation == 0), (observation == 1))), 0, 1
            ).astype(np.float32)

        else:
            encoded_observation = np.stack((
                (observation == -1),
                (observation == 0),
                (observation == 1)
            )).astype(np.float32)

        return encoded_observation

    def get_opponent_value(self, score):
        return -1*score

    def get_opponent_player(self, player):
        return -1*player

    def check_terminal_and_value(self, state, action):
        if self.is_position_a_winner(state, action):
            return (True, 1)
        if np.sum(self.get_valid_locations(state)) == 0:
            return (True, 0)
        return (False, 0)
