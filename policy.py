from Inleveropdracht_1.doolhof import Maze

from typing import List
import numpy as np


class Policy:
    def __init__(self, maze: Maze, values: List[int], gamma: int):
        self.maze = maze  # Our maze
        self.reward_matrix = np.reshape(values, (4, 4))[::-1]  # The reward matrix
        self.gamma = gamma
        self.policies = self.create_policy_grid()  # The policy grid

    def create_policy_grid(self):
        """Create the policy grid."""
        policy_grid = []
        for state in self.maze.states:
            if not state.is_endstate:
                policy_grid.append(['→', '←', '↑', '↓'])  # First every location gets a random policy
            else:
                policy_grid.append([None])
        return np.reshape(policy_grid, (4, 4))

    def get_surrounding_states(self, state):
        """Return the surrounding states of a state."""
        surrounding_states = []
        for direction in ['→', '←', '↑', '↓']:
            one_direction = Maze.step(self.maze, state.location, direction)
            surrounding_states.append(self.maze.states_matrix[one_direction.location[0]][one_direction.location[1]])
        return surrounding_states

    def bellman_equation(self, state):
        """Calculate the worth of a state using the Bellman equation."""
        reward = self.reward_matrix[state.location[0]][state.location[1]]
        value = state.value

        equation = reward + value * self.gamma
        return equation

    def select_action(self, original_state, surrounding_states, value_surrounding_states):
        """Based upon the surrounding states and their values, decide which policy is best. Return the highest value."""
        max_value = max(value_surrounding_states)

        max_value_states = []
        for state in range(len(value_surrounding_states)):
            if value_surrounding_states[state] == max_value:  # If a state has the highest value
                # Subtract the original state and his policy states and see which direction these states are
                difference = np.subtract(surrounding_states[state].location, original_state.location)

                if np.array_equal(difference, [0, 0]):
                    max_value_states.append('Itself')
                elif np.array_equal(difference, [1, 0]):
                    max_value_states.append('↑')
                elif np.array_equal(difference, [0, 1]):
                    max_value_states.append('→')
                elif np.array_equal(difference, [-1, 0]):
                    max_value_states.append('↓')
                elif np.array_equal(difference, [0, -1]):
                    max_value_states.append('←')

        self.policies[original_state.location[0]][original_state.location[1]] = max_value_states  # Update policy board
        return max_value
