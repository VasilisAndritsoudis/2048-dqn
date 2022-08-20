import os
from dqn import *
from constants import args


class Agent:
    def __init__(self):
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.eps_min = args.epsilon_min
        self.eps_dec = args.epsilon_decay
        self.lr = args.learning_rate
        self.action_space = [i for i in range(args.n_actions)]
        self.mem_size = args.max_memory_size
        self.batch_size = args.batch_size
        self.mem_counter = 0
        self.target_counter = 0
        self.target_replace_every = args.target_replace_every

        network = None
        memory = []
        if args.network_type == 'conv2d':
            network = Conv2dDQN
            if args.state_representation == 'one-hot':
                memory = np.zeros((self.mem_size, int(log2(args.win_tile)) + 1, *args.input_dims), dtype=np.float32)
            else:
                memory = np.zeros((self.mem_size, *args.input_dims), dtype=np.float32)
        elif args.network_type == 'linear':
            network = LinearDQN
            if args.state_representation == 'one-hot':
                memory = np.zeros((self.mem_size, np.prod(args.input_dims) * (int(log2(args.win_tile)) + 1)), dtype=np.float32)
            else:
                memory = np.zeros((self.mem_size, np.prod(args.input_dims)), dtype=np.float32)

        self.Q_eval = network()
        self.Q_target = network()
        self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.state_memory = memory.copy()
        self.new_state_memory = memory.copy()
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.win_memory = np.zeros(self.mem_size, dtype=np.bool)

    def get_model(self):
        return self.Q_eval

    def save_state(self, filename, epoch, aggr_ep_scores, aggr_ep_moves):
        state = {'epoch': epoch + 1,
                 'model_state_dict': self.Q_eval.state_dict(),
                 'model_optimizer': self.Q_eval.optimizer.state_dict(),
                 'target_state_dict': self.Q_target.state_dict(),
                 'target_optimizer': self.Q_target.optimizer.state_dict(),
                 'state_memory': self.state_memory,
                 'new_state_memory': self.new_state_memory,
                 'reward_memory': self.reward_memory,
                 'action_memory': self.action_memory,
                 'terminal_memory': self.terminal_memory,
                 'win_memory': self.win_memory,
                 'aggr_ep_scores': aggr_ep_scores,
                 'aggr_ep_moves': aggr_ep_moves
                 }
        T.save(state, filename)
        print("=> saved checkpoint '{}' (epoch {})".format(filename, epoch))

    def load_state(self, filename):
        episode = 0
        aggr_ep_scores = {'ep': [], 'avg': [], 'max': [], 'min': []}
        aggr_ep_moves = {'ep': [], 'avg': [], 'max': [], 'min': []}
        if os.path.isfile(filename):
            checkpoint = T.load(filename)
            episode = checkpoint['epoch']
            aggr_ep_scores = checkpoint['aggr_ep_scores']
            aggr_ep_moves = checkpoint['aggr_ep_moves']
            self.epsilon = self.epsilon - episode * self.eps_dec \
                if self.epsilon - episode * self.eps_dec > self.eps_min else self.eps_min
            self.Q_eval.load_state_dict(checkpoint['model_state_dict'])
            self.Q_eval.optimizer.load_state_dict(checkpoint['model_optimizer'])
            self.Q_target.load_state_dict(checkpoint['target_state_dict'])
            self.Q_target.optimizer.load_state_dict(checkpoint['target_optimizer'])
            self.state_memory = checkpoint['state_memory']
            self.new_state_memory = checkpoint['new_state_memory']
            self.reward_memory = checkpoint['reward_memory']
            self.action_memory = checkpoint['action_memory']
            self.terminal_memory = checkpoint['terminal_memory']
            self.win_memory = checkpoint['win_memory']
            print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return episode, aggr_ep_scores, aggr_ep_moves

    def store_transition(self, state, action, reward, state_, terminal, win):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal
        self.win_memory[index] = win

        self.mem_counter += 1

    def episode_done(self):
        self.target_counter += 1
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def choose_action(self, observation, ep_moves, avg_moves, use_epsilon):
        if use_epsilon:
            epsilon = self.epsilon
        else:
            epsilon = max(min(self.epsilon, ep_moves / avg_moves - 1), self.eps_min)

        if np.random.random() > epsilon:
            state = T.tensor(np.array([observation])).type(T.FloatTensor).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self, learn_iterations):
        if self.mem_counter < self.batch_size:
            return

        for _ in range(learn_iterations):
            max_mem = min(self.mem_counter, self.mem_size)

            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
            action_batch = self.action_memory[batch]
            reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            win_batch = T.tensor(np.logical_and(self.terminal_memory[batch], self.win_memory[batch]))\
                .to(self.Q_eval.device)
            lose_batch = T.tensor(np.logical_and(self.terminal_memory[batch], np.logical_not(self.win_memory[batch])))\
                .to(self.Q_eval.device)

            X = self.Q_eval.forward(state_batch)[batch_index, action_batch]
            q_next = self.Q_target.forward(new_state_batch)

            q_next[win_batch] = T.max(q_next).item() + 100
            q_next[lose_batch] = -100

            y = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

            self.Q_eval.optimizer.zero_grad()

            loss = self.Q_eval.loss(X, y).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()

        if self.target_counter > self.target_replace_every:
            self.target_counter = 0
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
