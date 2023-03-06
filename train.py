import time
import os

from torch.utils.tensorboard import SummaryWriter

from utils import *
from agent import Agent
from config import args
from environment import GameEngine


if __name__ == '__main__':
    writer = SummaryWriter(f"runs\\{args.network_type}\\{args.state_representation}")

    env = GameEngine()

    agent = Agent()

    ep_scores = []
    ep_moves = []

    aggr_scores = {'ep': [], 'avg': [], 'max': [], 'min': []}
    aggr_moves = {'ep': [], 'avg': [], 'max': [], 'min': []}
    aggr_wins = {'ep': [], 'win-rate': [], 'max-win-rate': []}
    aggr_time = []
    aggr_max_tile = []
    aggr_epsilon = []

    train_time = 0

    start_episode = 0

    max_tile = 0
    max_score = 0
    max_matrix = []

    wins = 0

    if os.path.exists(args.filepath):
        start_episode, aggr_scores, aggr_moves, aggr_wins, train_time = agent.load_state(args.filepath)

    time_keeper = time.time()

    for episode in range(start_episode, args.episodes + 1):
        ep_score = 0
        ep_move = 0

        win = False
        done = False
        invalid_moves = 0

        observation = env.reset()
        state = transform_state(np.array(observation))
        while not done:
            if args.epsilon_strategy_start == 0:
                # Custom strategy is not used
                action = agent.choose_action(state, ep_move, np.average(aggr_moves['avg']), False)
            else:
                if episode < args.episodes * args.epsilon_end * args.epsilon_strategy_start:
                    action = agent.choose_action(state, ep_move, np.average(aggr_moves['avg']), False)  # Explore early
                else:
                    # Custom strategy takes effect after epsilon strategy start episode is reached
                    action = agent.choose_action(state, ep_move, np.average(aggr_moves['avg']), True)  # Explore late

            observation_, reward, done, win, valid_move, tiles_moved = env.step(action)
            state_ = transform_state(np.array(observation_))

            ep_score += reward
            ep_move += 1

            if args.custom_reward:
                reward = env.reward(reward)
            agent_reward = log2(reward) if reward > 0 else 0
            agent_reward -= tiles_moved
            agent_reward = agent_reward - 20 if not valid_move else agent_reward

            if episode != 0:
                if episode % args.target_replace_every == 0:
                    win_per = wins / args.target_replace_every
                else:
                    win_per = wins / (episode % args.target_replace_every)
            else:
                win_per = 0

            agent_reward *= (1 - args.win_reward)
            agent_reward += agent_reward * args.win_reward * win_per

            if ep_move > args.max_episode_steps != 0:
                done = True

            if not valid_move:
                invalid_moves += 1
            else:
                invalid_moves = 0

            if invalid_moves > 10:
                done = True

            agent.store_transition(state, action, agent_reward, state_, done, win)
            state = state_
            observation = observation_

        agent.episode_done()
        agent.learn(args.learn_iterations)

        if win:
            wins += 1

        tile = np.max(observation)
        if tile > max_tile:
            max_tile = tile

        if ep_score > max_score:
            max_score = ep_score
            max_matrix = observation

        ep_scores.append(ep_score)
        ep_moves.append(ep_move)

        aggr_max_tile.append(max_tile)
        aggr_epsilon.append(agent.epsilon)

        aggr_time.append(train_time + time.time() - time_keeper)

        if episode % args.stats_every == 0 and episode != 0:
            # Update scores
            aggr_scores, avg_score, min_score, max_score = update_running_aggregates(
                aggr_scores, ep_scores, episode, args.stats_every)
            # Update moves
            aggr_moves, avg_moves, min_moves, max_moves = update_running_aggregates(
                aggr_moves, ep_moves, episode, args.stats_every)

            writer.add_scalars('Scores', {
                'Average': avg_score,
                'Min': min_score,
                'Max': max_score
            }, episode)

            writer.add_scalars('Moves', {
                'Average': avg_moves,
                'Min': min_moves,
                'Max': max_moves
            }, episode)

            writer.add_scalar('Epsilon', aggr_epsilon[-1], episode)

            writer.add_scalar('Max Tile', aggr_max_tile[-1], episode)

            writer.add_scalar('Train Time (seconds)', aggr_time[-1], episode)

            writer.close()

            time_str = time.strftime('%H:%M:%S', time.gmtime(aggr_time[-1]))

            if episode != 0:
                if episode % args.target_replace_every == 0:
                    win_per = wins / args.target_replace_every
                else:
                    win_per = wins / (episode % args.target_replace_every)
            else:
                win_per = 0

            print(f'Episode: {episode:>5d}, '
                  f'train time: {time_str}, '
                  f'win percentage: {win_per * 100:>3.1f}%, '
                  f'epsilon: {agent.epsilon:>1.2f}, '
                  f'average score: {avg_score:>7.1f}, '
                  f'average moves: {avg_moves:>5.1f}, '
                  f'max tile: {max_tile}, '
                  f'max score: {max_score}, '
                  f'max score matrix: \n{np.matrix(max_matrix)}')

            max_tile = 0
            max_score = 0

        if episode % args.target_replace_every == 0 and episode != 0:
            # Update wins
            aggr_wins['ep'].append(episode)
            aggr_wins['win-rate'].append(wins / args.target_replace_every)
            aggr_wins['max-win-rate'].append(np.max(aggr_wins['win-rate']))

            writer.add_scalars('Wins', {
                'Win Rate': wins / args.target_replace_every,
                'Max Win Rate': np.max(aggr_wins['win-rate']),
            }, episode)

            writer.close()

            wins = 0

        if episode % args.save_every == 0 and episode != 0:
            train_time += time.time() - time_keeper
            time_keeper = time.time()

            agent.save_state(args.filepath, episode, aggr_scores, aggr_moves, aggr_wins, train_time)
            plot(aggr_scores, aggr_moves, aggr_wins)
