from Inleveropdracht_2.assignment_1_1.doolhof import Maze
from Inleveropdracht_2.assignment_1_1.agent import Agent
from Inleveropdracht_2.assignment_1_1.policy import Policy


def setup():
    """Setup the environment."""
    #  Get a list of every possible coördinate
    matrix = []
    for i in range(0, 4):
        for j in range(0, 4):
            matrix.append([i, j])

    # Here are the predetermined rewards
    rewards = [-1, -1, -1, 40,
               -1, -1, -10, -10,
               -1, -1, -1, -1,
               10, -2, -1, -1
               ]

    values = [0 for i in range(len(rewards))]  # The values which is just a list of 16 zero's
    actions = ['→', '←', '↑', '↓']  # The actions. The arrows speak for the direction themselves
    endstates = [[0, 0], [3, 3]]  # The coördinates of the endstates
    gamma = 1  # The Gamma
    delta = 0.1  # The Delta
    location_agent = [1, 0]  # Location of the agent

    maze = Maze(matrix, values, actions, endstates)  # Initialize the maze
    policy = Policy(maze, rewards, gamma)  # Initialize the policy
    agent = Agent(maze, policy, location_agent, delta)  # Initialize the agent
    return maze, policy, agent


def main(y):
    """Choose which algorithm to use."""
    maze, policy, agent = setup()  # Setup the environment

    while True:  # Keep asking which algorithm the user wants to use until he makes a valid choice
        which_algorithm = input("Type '1' if you want to use the Monte Carlo Policy Evaluation algorithm.\n"
                            "And type '2' if you want to use the Temporal Difference Learning algorithm: \n")
        if which_algorithm == '1':
            return agent.monte_carlo_policy_evaluation(maze.states_matrix[2][0], y)
        elif which_algorithm == '2':
            return agent.temporal_difference_learning(maze.states_matrix[2][0], y)
        else:
            pass


main(1)
print('\n')
main(0.9)

