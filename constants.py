import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--network-type", type=str.lower, choices=['conv2d', 'linear'], default='conv2d', help="""Define 
    the neural network architecture""")
parser.add_argument("--state-representation", type=str.lower, choices=['log2', 'one-hot', 'as-is'], default='one-hot',
                    help="Define the state representation for the Neural Network")
parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate of the Neural Network")
parser.add_argument("--gamma", type=float, default=0.95, help="""Gamma [0, 1] is the discount factor. 
    It determines the importance of future rewards. A factor of 0 will make the agent consider only immediate reward, 
    a factor approaching 1 will make it strive for a long-term high reward""")
parser.add_argument("--epsilon", type=float, default=0.7, help="Exploration factor. [0, 1] for epsilon greedy train")
parser.add_argument("--epsilon-min", type=float, default=1e-3, help="Epsilon min value")
parser.add_argument("--epsilon-decay", type=float, default=(0.7 - 1e-3) / 10000, help="""Every step epsilon = epsilon - 
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
                                                    f"{args.network_type}_{args.state_representation}_"
                                                    f"lr={args.learning_rate}_"
                                                    f"eps={args.epsilon}_"
                                                    f"batch={args.batch_size}")

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
