import os
from math import log2
from torch.utils.tensorboard import SummaryWriter

from utils import *
from agent import Agent
from environment import GameEngine


if __name__ == '__main__':
    writer = SummaryWriter(f"runs\\{args.network_type}\\{args.state_representation}")

    env = GameEngine()

    agent = Agent(args=args)

    # writer.add_graph(agent.get_model())
    # writer.close()
    # sys.exit()

    # For stats
    ep_scores = []
    ep_moves = []
    aggr_ep_scores = {'ep': [], 'avg': [], 'max': [], 'min': []}
    aggr_ep_moves = {'ep': [], 'avg': [], 'max': [], 'min': []}

    start_episode = 0
    max_tile = 0
    max_score = 0
    max_matrix = []

    if os.path.exists(args.filepath):
        start_episode, aggr_ep_scores, aggr_ep_moves = agent.load_state(args.filepath)

    for episode in range(start_episode, args.episodes):
        ep_score = 0
        ep_move = 0

        done = False
        prev_action = None
        observation = env.reset()
        state = transform_state(np.array(observation))
        while not done:
            if episode < args.epsilon_strategy_start:
                action = agent.choose_action(state, ep_move, np.average(aggr_ep_moves['avg']), True)
            else:
                action = agent.choose_action(state, ep_move, np.average(aggr_ep_moves['avg']), False)

            observation_, reward, done, win, valid_move, tiles_moved = env.step(action)
            state_ = transform_state(np.array(observation_))

            ep_score += reward
            ep_move += 1

            # agent_reward = env.reward(reward)
            # agent_reward -= ep_move / MAX_EPISODE_MOVES
            #
            # agent_reward = agent_reward - 2 if not valid_move else agent_reward

            agent_reward = log2(reward) if reward > 0 else 0
            agent_reward -= tiles_moved
            agent_reward = agent_reward - 20 if not valid_move else agent_reward

            if ep_move > args.max_episode_steps:
                done = True

            if prev_action == action and not valid_move:
                done = True

            agent.store_transition(state, action, agent_reward, state_, done, win)
            state = state_
            observation = observation_

        agent.episode_done()
        agent.learn(args.learn_iterations)

        tile = np.max(observation)
        if tile > max_tile:
            max_tile = tile

        if ep_score > max_score:
            max_score = ep_score
            max_matrix = observation

        ep_scores.append(ep_score)
        ep_moves.append(ep_move)
        if not episode % args.stats_every:
            aggr_ep_scores, avg_reward, min_reward, max_reward = update_aggregates(
                aggr_ep_scores, ep_scores, episode, args.stats_every)
            aggr_ep_moves, avg_moves, min_moves, max_moves = update_aggregates(
                aggr_ep_moves, ep_moves, episode, args.stats_every)

            writer.add_scalars('Scores', {
                'Average': avg_reward,
                'Min': min_reward,
                'Max': max_reward
            }, episode)

            writer.add_scalars('Moves', {
                'Average': avg_moves,
                'Min': min_moves,
                'Max': max_moves
            }, episode)

            writer.add_scalar("Max Tile", max_tile, episode)
            writer.add_scalar("Epsilon", agent.epsilon, episode)

            writer.close()

            print(f'Episode: {episode:>5d}, average score: {avg_reward:>7.1f}, current epsilon: '
                  f'{agent.epsilon:>1.2f}, average moves: {avg_moves:>5.1f}, '
                  f'current max score: {max_score}, current max tile: {max_tile}, '
                  f'current max matrix: \n{np.matrix(max_matrix)}')

            max_tile = 0
            max_score = 0

        if not episode % args.save_every and episode != 0:
            agent.save_state(args.filepath, episode, aggr_ep_scores, aggr_ep_moves)
            plot(aggr_ep_scores, aggr_ep_moves)
