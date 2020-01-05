import numpy as np
from blackjack_env import Blackjack as env
from agent_q_extend import Agent_Q_extend as ag_q_extend
from agent_rule_based import Agent_rule_based as ag_rule_based
from agent_q import Agent_Q as ag_q
import matplotlib.pyplot as plt

deck_number = 10
num_episode = 1000


def plot_result(reward_list, name):
    plt.plot(reward_list)
    plt.title(name)
    plt.savefig("{}.png".format(name))
    plt.close()


def agent_Q_extend():
    blackjack = env(deck_number)
    agent = ag_q_extend(1, 0.2, 0.5, 0.1)
    reward_list = []

    for i in range(num_episode):
        current_state = blackjack.return_current_state()
        while True:
            action = agent.return_action(current_state)
            new_state, reward, is_done, new_round = blackjack.step(action)
            if is_done is True:
                break
            agent.add_replay_buffer(current_state, action, reward, new_state)
            if new_round is True:
                agent.train_replay_buffer()
            current_state = new_state
        reward_list.append(blackjack.total_reward)
        agent.update_epsilon_decay(i)
        blackjack.reset()
    agent.save_Q_table('q_table_q_extend.pkl')
    print("last reward is {}".format(reward_list[-1]))
    plot_result(reward_list, "Agent Q Extend")


def agent_Q():
    blackjack = env(deck_number)
    agent = ag_q(1, 0.2, 1, 0.1)
    reward_list = []

    for i in range(num_episode):
        current_state = blackjack.return_current_state()
        while True:
            action = agent.return_action(current_state)
            new_state, reward, is_done, new_round = blackjack.step(action)
            if is_done is True:
                break
            agent.update_Q_table(current_state, action, reward, new_state)
            current_state = new_state
        reward_list.append(blackjack.total_reward)
        agent.update_epsilon_decay(i)
        blackjack.reset()
    agent.save_Q_table('q_table_q.pkl')
    print("last reward is {}".format(reward_list[-1]))
    plot_result(reward_list, "Agent Q")


def agent_rule_based():
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
    print("last reward is {}".format(reward_list[-1]))
    plot_result(reward_list, "Agent rule based")


if __name__ == '__main__':
    agent_Q_extend()
    agent_Q()
    agent_rule_based()
