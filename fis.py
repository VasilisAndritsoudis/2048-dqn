import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import torch as T
import os
import matplotlib.pyplot as plt


def plot(aggr_ep_scores, aggr_ep_moves):
    plt.plot(aggr_ep_scores['ep'], aggr_ep_scores['avg'], label="average scores")
    plt.plot(aggr_ep_scores['ep'], aggr_ep_scores['max'], label="max scores")
    plt.plot(aggr_ep_scores['ep'], aggr_ep_scores['min'], label="min scores")
    plt.legend(loc='upper left')
    plt.show()

    plt.plot(aggr_ep_moves['ep'], aggr_ep_moves['avg'], label="average moves")
    plt.plot(aggr_ep_moves['ep'], aggr_ep_moves['max'], label="max moves")
    plt.plot(aggr_ep_moves['ep'], aggr_ep_moves['min'], label="min moves")
    plt.legend(loc='upper left')
    plt.show()


filename = 'States\\big_memory_custom_reward.dqn'

# Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
aggr_ep_scores = {'ep': [], 'avg': [], 'max': [], 'min': []}
aggr_ep_moves = {'ep': [], 'avg': [], 'max': [], 'min': []}
if os.path.isfile(filename):
    checkpoint = T.load(filename)
    aggr_ep_scores = checkpoint['aggr_ep_scores']
    aggr_ep_moves = checkpoint['aggr_ep_moves']
    print("=> loaded checkpoint '{}'".format(filename))
else:
    print("=> no checkpoint found at '{}'".format(filename))

score = ctrl.Antecedent(np.arange(0, 25000, 1), 'score')
moves = ctrl.Antecedent(np.arange(0, 1000, 1), 'moves')
performance = ctrl.Consequent(np.arange(0, 100, 1), 'performance')

score['low'] = fuzz.zmf(score.universe, 0, 7500)
score['medium'] = fuzz.gbellmf(score.universe, 5000, 4, 10000)
score['high'] = fuzz.smf(score.universe, 15000, 25000)
score.view()

moves['few'] = fuzz.trimf(moves.universe, [0, 0, 300])
moves['average'] = fuzz.trimf(moves.universe, [200, 500, 800])
moves['many'] = fuzz.trimf(moves.universe, [700, 1000, 1000])
moves.view()

performance['bad'] = fuzz.zmf(performance.universe, 0, 30)
performance['below-average'] = fuzz.pimf(performance.universe, 20, 35, 40, 65)
performance['above-average'] = fuzz.gbellmf(performance.universe, 7.5, 1.5, 62.5)
performance['good'] = fuzz.sigmf(performance.universe, 85, 0.4)
performance.view()

rule1 = ctrl.Rule(score['low'], consequent=performance['bad'])
rule2 = ctrl.Rule(score['medium'] & moves['many'], consequent=performance['below-average'])
rule3 = ctrl.Rule(score['medium'] & moves['average'], consequent=performance['below-average'])
rule4 = ctrl.Rule(score['medium'] & moves['few'], consequent=performance['above-average'])
rule5 = ctrl.Rule(score['high'] & moves['many'], consequent=performance['above-average'])
rule6 = ctrl.Rule(score['high'] & moves['average'], consequent=performance['good'])
rule7 = ctrl.Rule(score['high'] & moves['few'], consequent=performance['good'])

performance_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
performance_fis = ctrl.ControlSystemSimulation(performance_ctrl)

plot(aggr_ep_scores, aggr_ep_moves)
performance_fis.input['score'] = np.average(aggr_ep_scores["avg"])
performance_fis.input['moves'] = np.average(aggr_ep_moves["avg"])
# print(f'avg_score: {performance_fis.input["score"]}, avg_moves: {performance_fis.input["moves"]}')

performance_fis.compute()

print(performance_fis.output['performance'])
performance.view(sim=performance_fis)
