from Inleveropdracht_1.doolhof import Maze
from Inleveropdracht_1.agent import Agent
from Inleveropdracht_1.policy import Policy


def main():
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

    # Start the value iteration
    # agent.monte_carlo_policy_evaluation(maze.states_matrix[2][0])
    agent.value_iteration()
    return None
    #
    # print('\n')
    # if input("Ben je klaar om de agent het pad af te zien lopen? \n"
    #          "Zeg 'Ja' als je hier klaar voor bent: ") == 'Ja':
    #     agent.agent_path()


main()
