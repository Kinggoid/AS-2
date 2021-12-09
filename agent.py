from Inleveropdracht_2.assignment_1_1.doolhof import Maze
from Inleveropdracht_2.assignment_1_1.policy import Policy

import random
from typing import List
from statistics import mean


class Agent:
    def __init__(self, maze: Maze, policy: Policy, location: List[int], delta: float):
        self.maze = maze  # Our maze
        self.location = location  # Location of the agent
        self.delta = delta  # A number to indicate how little the difference between two value iterations has to be
        # in order for the iterations to stop
        self.policy = policy  # Our policy

    def check_difference_in_number(self, state):
        """Check the difference between two numbers."""
        new_value = abs(state.new_value)
        value = abs(state.value)
        return abs(new_value - value)

    def delta_check(self):
        """Check whether any states have changes since the last iteration."""
        for state in self.maze.states:
            if self.check_difference_in_number(state) > self.delta:
                return False
        return True

    def print_policies(self):
        """Print the current policies of our value iteration."""
        for i in self.policy.policies[::-1]:
            row = []
            for j in i:
                row.append(j)
            print(row)

    def nieuwe_value_matrix(self):
        """Print the current values of our value iteration."""
        values = []
        for i in self.maze.states_matrix[::-1]:
            row = []
            for j in i:
                row.append(j.new_value)
                j.value = j.new_value  # Update the values of the states to their new values
            print(row)
            values.append(row)
        return values

    def list_to_4x4_matrix(self, lst):
        lst = list(lst)
        matrix = []
        for i in range(len(lst) - 4, -4, -4):
            row = []
            for j in range(4):
                row.append(lst[i + j])
            print(row)
            matrix.append(row)
        return matrix

    def value_iteration(self):
        """The value iteration main loop."""
        k = 1  # Amount of iterations

        while True:
            print('Iteration ' + str(k))
            k += 1

            for state in self.maze.states:  # For every state
                if state.is_endstate:  # If a state is an endstate, we ignore it. Their value cannot go up.
                    continue

                surrounding_states = Policy.get_surrounding_states(self.policy, state)  # Get surrounding states

                value_surrounding_states = []  # Get the value of these surrounding states
                for surrounding_state in surrounding_states:
                    value_surrounding_states.append(Policy.bellman_equation(self.policy, surrounding_state))

                # Update the policies and get the best value of this best action
                new_value = Policy.select_action(self.policy, state, surrounding_states, value_surrounding_states)

                # The state's new value is the value of his best choice (value iteration)
                state.new_value = new_value

            if self.delta_check():  # The value iteration has come to a stop
                print('These are the new values of the matrix after the value iteration.')
                self.nieuwe_value_matrix()

                print('\n')

                print('These are the new policies after the value iteration.')
                self.print_policies()
                break

            print('These are the new values of the matrix after the value iteration.')
            self.nieuwe_value_matrix()

            print('\n')

            print('These are the new policies after the value iteration.')
            self.print_policies()

            print('\n')

    def episode(self, startstate, policy):
        """Create an episode."""
        path = []
        state = startstate

        while not state.is_endstate:  # If the new state isn't an endstate, save it and get a new state from the policy
            path.append(state)
            action = random.choice(policy.policies[state.location[0]][state.location[1]])
            state = self.maze.step(state.location, action)
        path.append(state)
        return path

    def initialize_values_and_returns(self):
        """Initialize the value and returns dictionaries."""
        values = {}
        returns = {}
        for state in self.maze.states:
            if state.is_endstate:
                values[str(state)] = self.policy.reward_matrix[state.location[0]][state.location[1]]
            else:
                values[str(state)] = 0
            returns[str(state)] = []
        return returns, values

    def monte_carlo_policy_evaluation_main_loop(self, beginstate, gamma):
        """Monte Carlo main loop."""
        returns, values = self.initialize_values_and_returns()

        k = 0
        amount_of_episodes = 5000  # We run 5000 episodes
        while k != amount_of_episodes:
            k += 1
            y = gamma
            episode = self.episode(beginstate, self.policy)  # Create episode
            episode.reverse()  # Since we are going to be working backwards, we reverse the episode list so we can
                                # iterate starting from the end
            g = 0

            for state in range(1, len(episode)):  # For every state in the episode
                next_state = episode[state - 1]  # What state would follow this state in the episode
                state_location = next_state.location
                next_state_reward = self.policy.reward_matrix[state_location[0], state_location[1]]
                g = y * g + next_state_reward

                if episode[state] not in episode[:state]:  # Only update the last states in the episode
                    returns[str(episode[state])].append(g)
                    return_mean = mean(returns[str(episode[state])])
                    values[str(episode[state])] = return_mean
        return returns, values

    def monte_carlo_policy_evaluation(self, beginstate, gamma):
        """Monte carlo policy evaluation main."""
        returns, values = self.monte_carlo_policy_evaluation_main_loop(beginstate, gamma)  # The main loop

        print('These are the values: ')
        self.list_to_4x4_matrix(values.values())
        return values

    def initialize_values(self):
        """Initialize values."""
        values = {}
        for state in self.maze.states:
            if state.is_endstate:
                values[str(state)] = [self.policy.reward_matrix[state.location[0]][state.location[1]]]
            else:
                values[str(state)] = [0]
        return values

    def TDL_print_grid(self, lst):
        """Print the grid for the Temporal Difference Learning algorithm."""
        lst = list(lst)
        matrix = []
        for i in range(len(lst) - 4, -4, -4):
            row = []
            for j in range(4):
                row.append(mean(lst[i + j]))
            print(row)
            matrix.append(row)
        return matrix

    def temporal_difference_learning_main_loop(self, beginstate, values, gamma):
        """The temporal difference learning main loop."""
        k = 0
        amount_of_episodes = 5000
        while k != amount_of_episodes:
            k += 1
            state = beginstate

            while not state.is_endstate:  # While we haven't reached the endstate
                action = random.choice(self.policy.policies[state.location[0]][state.location[1]])  # Pick an action
                next_state = self.maze.step(state.location, action)  # Get the next state
                value = values[str(state)][-1]  # Last state value
                next_state_value = values[str(next_state)][-1]  # Last value of next state
                next_state_reward = self.policy.reward_matrix[state.location[0]][state.location[1]]  # Next state reward
                alpha = 1
                new_value = value + alpha * (next_state_reward + (gamma * next_state_value) - value)
                values[str(state)].append(new_value)
                state = next_state  # Continue with next state
        return values

    def temporal_difference_learning(self, beginstate, gamma):
        """Temporal Difference Learning main function."""
        values = self.initialize_values()  # Initialize values

        values = self.temporal_difference_learning_main_loop(beginstate, values, gamma)  # Main loop

        print('These are the values: ')
        self.TDL_print_grid(values.values())  # Print values/results
        return values

    def agent_path(self):
        """Simulate an agent going through the maze and taking the value iterations best path."""
        k = 1  # How many steps are necesarry for the Agent to find the endstate

        while True:
            k += 1
            policy = self.policy.policies[self.location[1]][self.location[0]]  # Find the policy
            new_state = self.maze.step(self.location,
                                       random.choice(policy))  # Find the coordinates of our next location

            self.location = new_state.location
            print('The agent is currently on coördinates: ' + str(self.location))

            if new_state.is_endstate:  # If we land on an endstate, we stop the simulation
                break

        print("..." + str(k) + " steps for the agent to get to the endstate.")
