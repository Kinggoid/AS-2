from dataclasses import dataclass

from Inleveropdracht_1.doolhof import Maze
from Inleveropdracht_1.policy import Policy

import random
from typing import List
import numpy as np
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
        lst = list(lst)[::-1]
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
        path = []

        state = startstate
        while not state.is_endstate:
            path.append(state)
            action = random.choice(policy.policies[state.location[0]][state.location[1]])
            state = self.maze.step(state.location, action)
        return path

    # def unique(self, lst):
    #     output = []
    #     for x in lst:
    #         if x not in output:
    #             output.append(x)
    #     return output

    def monte_carlo_policy_evaluation(self, beginstate):
        values = {}
        returns = {}
        for state in self.maze.states:
            values[str(state)] = 0
            returns[str(state)] = []

        k = 0

        while k != 500:
            k += 1
            episode = self.episode(beginstate, self.policy)
            episode.reverse()
            g = 0
            # episode[len(episode) - 1 :: -1]

            for state in range(1, len(episode)):
                next_state = episode[state - 1]
                state_location = next_state.location
                g += self.policy.reward_matrix[state_location[0], state_location[1]]

                if episode[state] not in episode[state + 1:]:
                    returns[str(episode[state])].append(g)
                    return_mean = mean(returns[str(episode[state])])
                    values[str(episode[state])] = return_mean

        for state in range(len(self.maze.states)):
            self.maze.states[state].value = values[str(self.maze.states[state])]

        for state in range(len(self.maze.states)):
            surrounding_states = Policy.get_surrounding_states(self.policy,
                                                               self.maze.states[state])  # Get surrounding states

            value_surrounding_states = []  # Get the value of these surrounding states
            for surrounding_state in surrounding_states:
                value_surrounding_states.append(Policy.bellman_equation(self.policy, surrounding_state))

            # Update the policies and get the best value of this best action
            Policy.select_action(self.policy, self.maze.states[state], surrounding_states, value_surrounding_states)

        self.list_to_4x4_matrix(values.values())
        return values

    def agent_path(self):
        """Simulate an agent going through the maze and taking the value iterations best path."""
        k = 1  # How many steps are necesarry for the Agent to find the endstate

        while True:
            k += 1
            policy = self.policy.policies[self.location[1]][self.location[0]]  # Find the policy
            new_state = self.maze.step(self.location,
                                       random.choice(policy))  # Find the coordinates of our next location

            location = new_state.location
            print('The agent is currently on coördinates: ' + str(location))

            if new_state.is_endstate:  # If we land on an endstate, we stop the simulation
                break

        print("..." + str(k) + " steps for the agent to get to the endstate.")
