import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import log2

parser = argparse.ArgumentParser()
parser.add_argument("--network-type", type=str.lower, choices=['conv2d', 'linear'], default='conv2d', help="""Define 
    the neural network architecture""")
parser.add_argument("--state-representation", type=str.lower, choices=['log2', 'one-hot', 'as-is'], default='one-hot',
                    help="Define the state representation for the Neural Network")
parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate of the Neural Network")
parser.add_argument("--gamma", type=float, default=0.95, help="""Gamma [0, 1] is the discount factor. 
    It determines the importance of future rewards. A factor of 0 will make the agent consider only immediate reward, 
    a factor approaching 1 will make it strive for a long-term high reward""")
parser.add_argument("--epsilon", type=float, default=1, help="Exploration factor. [0, 1] for epsilon greedy train")
parser.add_argument("--epsilon-min", type=float, default=1e-3, help="Epsilon min value")
parser.add_argument("--epsilon-decay", type=float, default=(1 - 1e-3) / 10000, help="""Every step epsilon = epsilon - 
    decay, in order to decrease constantly""")
parser.add_argument("--batch-size", type=int, default=1024, help="Size of the batch used in the training step")
parser.add_argument("--input-dims", type=tuple, default=(4, 4), help="Size of the input dimensions")
parser.add_argument("--n_actions", type=int, default=4, help="Number of allowed actions")
parser.add_argument("--max-memory-size", type=int, default=100000, help="Size of history memory")
parser.add_argument("--target-replace-every", type=int, default=250, help="Target network update frequency (episodes)")
parser.add_argument("--max-episode-steps", type=int, default=1500, help="Max number of steps per episode")
parser.add_argument("--epsilon-strategy-start", type=int, default=0, help="""Start episode of the custom epsilon decay 
    strategy. If start == 0 then custom strategy is not used""")
parser.add_argument("--learn-iterations", type=int, default=10, help="Number of training iterations after each episode")
parser.add_argument("--stats-every", type=int, default=100, help="Print training statistics frequency")
parser.add_argument("--save-every", type=int, default=500, help="Network state save frequency")
parser.add_argument("--win-tile", type=int, default=2048, help="Win condition tile")
parser.add_argument("--episodes", type=int, default=50000, help="Number of training episodes")

args = parser.parse_known_args()[0]

parser.add_argument("--filepath", type=str, default=f"states\\{args.network_type}\\{args.state_representation}\\"
                                                    f"{args.network_type}_{args.state_representation}")

args = parser.parse_args()


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
