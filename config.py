import argparse
import hashlib

parser = argparse.ArgumentParser(description="DQN agent parameters",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--network-type", type=str.lower, choices=['conv2d', 'linear'], default='conv2d', help="""Define 
    the neural network architecture""")
parser.add_argument("--state-representation", type=str.lower, choices=['log2', 'one-hot', 'as-is'], default='one-hot',
                    help="Define the state representation for the Neural Network")
parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate of the Neural Network")
parser.add_argument("--gamma", type=float, default=0.99, help="""Gamma [0, 1] is the discount factor. 
    It determines the importance of future rewards. A factor of 0 will make the agent consider only immediate reward, 
    a factor approaching 1 will make it strive for a long-term high reward""")
parser.add_argument("--epsilon", type=float, default=0.75, help="Exploration factor. [0, 1] for epsilon greedy train")
parser.add_argument("--epsilon-min", type=float, default=1e-3, help="Epsilon min value")
parser.add_argument("--epsilon-end", type=float, default=0.2, help="""The percentage of episodes it takes for epsilon 
    to reach epsilon-min. Every step epsilon = epsilon - decay, in order to decrease constantly""")
parser.add_argument("--batch-size", type=int, default=2048, help="Size of the batch used in the training step")
parser.add_argument("--input-dims", type=tuple, default=(4, 4), help="Size of the input dimensions")
parser.add_argument("--n_actions", type=int, default=4, help="Number of allowed actions")
parser.add_argument("--max-memory-size", type=int, default=250000, help="Size of history memory")
parser.add_argument("--target-replace-every", type=int, default=250, help="""Target network update frequency (episodes).
    It also affects the win rate (wr = wins / (episode % target_replace_every)""")
parser.add_argument("--max-episode-steps", type=int, default=2000, help="""Max number of steps per episode. 
    If max-episode-steps == 0 then there is not limit to the number of steps per episode""")
parser.add_argument("--epsilon-strategy-start", type=float, default=0.25, help="""The percentage of epsilon episodes 
    (epsilon is not min) it takes for the custom epsilon decay strategy to start. If epsilon-strategy-start == 0 then 
    custom strategy is not used. Custom epsilon decay strategy: Begin with small epsilon in the early game and then 
    gradually increase epsilon when going further into the game. The gradual increase is based on the average moves the
    agent makes at each point of the training. The idea of the custom strategy is that the early game has already been
    explored, thus there is no point in exploring it further and it is more beneficial to explore the late game 
    states""")
parser.add_argument("--learn-iterations", type=int, default=10, help="Number of training iterations after each episode")
parser.add_argument("--stats-every", type=int, default=100, help="Print training statistics frequency")
parser.add_argument("--save-every", type=int, default=500, help="Network state save frequency")
parser.add_argument("--win-tile", type=int, default=2048, help="Win condition tile")
parser.add_argument("--custom-reward", type=bool, default=False, help="""Use custom reward for training. Custom reward: 
    Penalizes the distance of the highest value tile from the top left corner. It also penalizes the distances between 
    close tile values""")
parser.add_argument("--penalty", type=float, default=0.1, help="""Percentage of custom reward affected by strategy 
    penalty. Only used when custom-reward is activated""")
parser.add_argument("--win-reward", type=float, default=0.75, help="Percentage of reward affected by win rate")
parser.add_argument("--episodes", type=int, default=25000, help="Number of training episodes")

args = parser.parse_known_args()[0]

filename = f"{args.network_type}_{args.state_representation}_" \
    f"learning_rate={args.learning_rate}_" \
    f"gamma={args.gamma}_" \
    f"epsilon={args.epsilon}_" \
    f"epsilon-min={args.epsilon_min}_" \
    f"epsilon-end={args.epsilon_end}" \
    f"batch-size={args.batch_size}_" \
    f"max-memory-size={args.max_memory_size}_" \
    f"target-replace-every={args.target_replace_every}_" \
    f"max-episode-steps={args.max_episode_steps}_" \
    f"epsilon-strategy-start={args.epsilon_strategy_start}_" \
    f"learn-iterations={args.learn_iterations}_" \
    f"win-tile={args.win_tile}_" \
    f"custom-reward={args.custom_reward}_" \
    f"penalty={args.penalty}_" \
    f"win-reward={args.win_reward}_" \
    f"stats-every={args.stats_every}_" \
    f"save-every={args.save_every}_" \
    f"episodes={args.episodes}"

filename = hashlib.sha256(filename.encode('utf-8')).hexdigest()

print("Config hash: " + filename)

parser.add_argument("--filepath", type=str,
                    default=f"states\\{args.network_type}\\{args.state_representation}\\{filename}.pt")

parser.add_argument("--complete-filepath", type=str,
                    default=f"states\\best\\best.pt")

decay = (args.epsilon - args.epsilon_min) / (args.episodes * args.epsilon_end)

parser.add_argument("--epsilon-decay", type=float, default=decay)

args = parser.parse_args()

SIZE = 400
GRID_LEN = args.input_dims[0]
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"

BACKGROUND_COLOR_DICT = {
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
    4096: "#eee4da",
    8192: "#edc22e",
    16384: "#f2b179",
    32768: "#f59563",
    65536: "#f67c5f",
}

CELL_COLOR_DICT = {
    2: "#776e65",
    4: "#776e65",
    8: "#f9f6f2",
    16: "#f9f6f2",
    32: "#f9f6f2",
    64: "#f9f6f2",
    128: "#f9f6f2",
    256: "#f9f6f2",
    512: "#f9f6f2",
    1024: "#f9f6f2",
    2048: "#f9f6f2",
    4096: "#776e65",
    8192: "#f9f6f2",
    16384: "#776e65",
    32768: "#776e65",
    65536: "#f9f6f2",
}

FONT = ("Verdana", 40, "bold")

KEY_QUIT = "Escape"
KEY_BACK = "b"
KEY_DQN = "w"

KEY_UP = "Up"
KEY_DOWN = "Down"
KEY_LEFT = "Left"
KEY_RIGHT = "Right"

KEY_UP_ALT1 = "w"
KEY_DOWN_ALT1 = "s"
KEY_LEFT_ALT1 = "a"
KEY_RIGHT_ALT1 = "d"

KEY_UP_ALT2 = "i"
KEY_DOWN_ALT2 = "k"
KEY_LEFT_ALT2 = "j"
KEY_RIGHT_ALT2 = "l"
