import numpy as np
from blackjack_env import Blackjack as env
from agent_q_extend import Agent_Q_Extend as ag_q_extend
from agent_rule_based import Agent_rule_based as ag_rule_based
from agent_q import Agent_Q as ag_q
import matplotlib.pyplot as plt
import pandas as pd

deck_number = 10  # number of deck shuffle together
num_episode = 1000  # number of episode


def plot_result(reward_lists):
    """
    plotting rewards agent got on each episode
    """
    for reward_list in reward_lists:
        plt.plot(reward_list)
    plt.title("Reward on each episode")
    plt.savefig("{}.png".format("Reward on each episode"))
    plt.show()
    plt.close()


def print_q_table(path):
    """
    printing Q table of agent showing greedy action agent will take on each state
    """
    Q_table = np.load(path)
    possible_state = [i for i in range(2, 22)]
    possible_action = ['h', 's']
    possible_dis = ['e low cards', 'more low cards', 'same amount', 'more high cards', 'e high cards']
    pd_Q_table = pd.DataFrame(np.zeros((20, 5)), columns=possible_dis, index=possible_state)

    for val_index, i in enumerate(possible_state):
        for dis_index, j in enumerate(possible_dis):
            pd_Q_table[j][i] = possible_action[np.argmax(Q_table[val_index, dis_index, :])]

    print(pd_Q_table)


def agent_Q_extend():
    """
    Running Agent Q with Q-learning with extra capability on Blackjack environment
    :return: list of reward on each episode
    """
    blackjack = env(deck_number)
    agent = ag_q_extend(1, 0.2, 0.2, 0.01)
    reward_list = []

    for i in range(num_episode):
        current_state = blackjack.return_current_state()
        while True:
            action = agent.return_action(current_state)
            new_state, reward, is_done, new_round = blackjack.step(action)
            if is_done is True:
                break
            agent.update_Q_table(current_state, action, reward, new_state, new_round)  # update state-action pair
            agent.add_replay_buffer(current_state, action, reward, new_state, new_round)
            agent.train_replay_buffer()
            current_state = new_state
        reward_list.append(blackjack.total_reward)
        agent.update_epsilon_decay(i)
        blackjack.reset()
    agent.save_Q_table('q_table_q_extend.npy')
    print("last reward for Agent Q Extend is {}".format(reward_list[-1]))
    return reward_list


def agent_Q():
    """
    Running Agent Q with basic Q-learning capability on Blackjack environment
    :return: list of reward on each episode
    """
    blackjack = env(deck_number)
    agent = ag_q(1, 0.2, 0.2, 0.01)
    reward_list = []

    for i in range(num_episode):
        current_state = blackjack.return_current_state()
        while True:
            action = agent.return_action(current_state)
            new_state, reward, is_done, new_round = blackjack.step(action)
            if is_done is True:
                break
            agent.update_Q_table(current_state, action, reward, new_state)  # update state-action pair
            current_state = new_state
        reward_list.append(blackjack.total_reward)
        agent.update_epsilon_decay(i)
        blackjack.reset()
    agent.save_Q_table('q_table_q.npy')
    print("last reward for Agent Q is {}".format(reward_list[-1]))
    return reward_list


def agent_rule_based():
    """
    Running Agent rule based which only perform 'hit' when card value is below 12 on Blackjack environment
    and 'stick' otherwise
    :return: list of reward on each episode
    """
    blackjack = env(deck_number)
    agent = ag_rule_based()
    reward_list = []

    for i in range(num_episode):
        current_state = blackjack.return_current_state()
        while True:
            action = agent.return_action(current_state)
            new_state, reward, is_done, new_round = blackjack.step(action)
            if is_done is True:
                break
            current_state = new_state
        reward_list.append(blackjack.total_reward)
        blackjack.reset()
    print("last reward for Agent rule based is {}".format(reward_list[-1]))
    return reward_list


if __name__ == '__main__':
    agent_q_extend_rl = agent_Q_extend()  # training Agent Q Extend
    agent_q_rl = agent_Q()  # training Agent Q
    agent_rule_based_rl = agent_rule_based()  # training Agent rule based
    plot_result([agent_q_extend_rl, agent_q_rl, agent_rule_based_rl])  # plot agent's rewards on each episode
    print_q_table("q_table_q_extend.npy")  # print Q table of agent showing greedy action agent will take on each state
