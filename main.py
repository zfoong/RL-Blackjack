import numpy as np
from blackjack_env import Blackjack as env
from agent import Agent as ag
import matplotlib.pyplot as plt


def main():
    deck_number = 5
    blackjack = env(deck_number)
    num_episode = 1000
    agent = ag(1, 0.1, 0.90)
    reward_list = []

    for i in range(num_episode):
        current_state = blackjack.return_current_state()
        current_state = blackjack.return_state_id(current_state)
        while True:
            action = agent.return_action(current_state)
            converted_action = blackjack.possible_action[action]
            new_state, reward, is_done = blackjack.step(converted_action)
            if is_done is True:
                break
            new_state = blackjack.return_state_id(new_state)
            print("State : {}, Reward : {}, is done : {}".format(new_state, reward, is_done))
            agent.update_Q_table(current_state, action, reward, new_state)
            current_state = new_state
        reward_list.append(blackjack.total_reward)
        agent.update_epsilon_decay(i)
        blackjack.reset()
    agent.print_Q_table()
    print("agent epsilon = {}".format(agent.epsilon))
    plt.plot(reward_list)
    plt.show()


if __name__ == '__main__':
    main()
