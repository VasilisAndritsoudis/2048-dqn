import numpy as np
import matplotlib.pyplot as plt
from math import log2

from constants import args


def transform_state(state):
    if args.state_representation == 'log2':
        state[state == 0] = 1
        state = np.log2(state)
    elif args.state_representation == 'one-hot':
        state = np.reshape(state, -1)
        state[state == 0] = 1
        state = np.log2(state)
        state = state.astype(int)
        state = np.eye(int(log2(args.win_tile)) + 1)[state]

        if args.network_type == 'conv2d':
            state_ = np.empty([int(log2(args.win_tile)) + 1, *args.input_dims])
            for i in range(state.shape[1]):
                for j in range(state.shape[0]):
                    state_[i][j % args.input_dims[0]][j // args.input_dims[0]] = state[j][i]
            state = state_

    if args.network_type == 'linear':
        state = np.reshape(state, (-1))

    return state


def update_aggregates(aggr, ep_values, episode, update_every):
    avg = sum(ep_values[-update_every:]) / update_every
    min_val = min(ep_values[-update_every:])
    max_val = max(ep_values[-update_every:])
    aggr['ep'].append(episode)
    aggr['avg'].append(avg)
    aggr['max'].append(max(ep_values[-update_every:]))
    aggr['min'].append(min(ep_values[-update_every:]))
    return aggr, avg, min_val, max_val


def plot(aggr_ep_scores, aggr_ep_moves):
    plt.plot(aggr_ep_scores['ep'], aggr_ep_scores['avg'], label="average scores")
    plt.plot(aggr_ep_scores['ep'], aggr_ep_scores['max'], label="max scores")
    plt.plot(aggr_ep_scores['ep'], aggr_ep_scores['min'], label="min scores")
    plt.legend()
    plt.show()

    plt.plot(aggr_ep_moves['ep'], aggr_ep_moves['avg'], label="average moves")
    plt.plot(aggr_ep_moves['ep'], aggr_ep_moves['max'], label="max moves")
    plt.plot(aggr_ep_moves['ep'], aggr_ep_moves['min'], label="min moves")
    plt.legend()
    plt.show()
