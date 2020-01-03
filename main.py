import numpy as np
from blackjack_env import Blackjack as env
from agent import Agent as ag
from agent_2 import Agent as ag2
import matplotlib.pyplot as plt


def main():
    deck_number = 200
    blackjack = env(deck_number)
    num_episode = 1000
    agent = ag(1, 0.2, 1)
    reward_list = []

    for i in range(num_episode):
        current_state = blackjack.return_current_state()
        while True:
            action = agent.return_action(current_state)
            new_state, reward, is_done = blackjack.step(action)
            if is_done is True:
                break
            # print("State : {}, Reward : {}, is done : {}".format(new_state, reward, is_done))
            agent.update_Q_table(current_state, action, reward, new_state)
            current_state = new_state
        reward_list.append(blackjack.total_reward)
        agent.update_epsilon_decay(i)
        blackjack.reset()
    agent.print_Q_table()
    agent.save_Q_table('q_table.pkl')
    # print("agent epsilon = {}".format(agent.epsilon))
    print("last reward is {}".format(reward_list[-1]))
    plt.plot(reward_list)
    plt.title("Agent")
    #plt.show()


def main_test():
    deck_number = 200
    blackjack = env(deck_number)
    num_episode = 1000
    agent = ag2()
    reward_list = []

    for i in range(num_episode):
        current_state = blackjack.return_current_state()
        while True:
            action = agent.return_action(current_state)
            new_state, reward, is_done = blackjack.step(action)
            if is_done is True:
                break
            # print("State : {}, Reward : {}, is done : {}".format(new_state, reward, is_done))
            current_state = new_state
        reward_list.append(blackjack.total_reward)
        blackjack.reset()
    print("last reward is {}".format(reward_list[-1]))
    plt.plot(reward_list)
    plt.title("Agent 2")
    #plt.show()


if __name__ == '__main__':
    main()
    main_test()
