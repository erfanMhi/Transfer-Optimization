import numpy as np
import argparse
import os

from copy import deepcopy
from time import time
from pprint import pprint
from utils.data_manipulators import *
from evolution.operators import *
from to.probabilistic_model import ProbabilisticModel
from to.mixture_model import MixtureModel
from evolution.chromosome import *


def evolutionary_algorithm(problem,
                           dims,
                           psize=100,
                           gen=100,
                           src_models=[],
                           stop_condition=True,
                           create_model=True):

    fitness_hist = np.zeros((gen, psize))
    fitness_time = np.zeros((gen))

    init_func = lambda n: np.round(np.random.rand(n))
    pop = get_pop_init(psize, dims, init_func)
    start = time()
    for i in range(psize):
        pop[i].fitness_calc(problem)

    bestfitness = np.max(pop).fitness
    fitness = Chromosome.fitness_to_numpy(pop)
    fitness_hist[0, :] = fitness

    fitness_time[0] = time() - start
    counter = 0  # Generation Repetition without fitness improvement counter
    for i in range(1, gen):
        start = time()

        # Crossover & Mutation
        offsprings = total_crossover(pop)
        for j in range(psize):
            offsprings[j].mutation(1 / dims)

        # Fitness Calculation
        cfitness = np.zeros(psize)
        for j in range(psize):
            cfitness[j] = offsprings[j].fitness_calc(problem)

        # Selection
        pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                       np.concatenate((fitness, cfitness)),
                                       psize)

        fitness_hist[i, :] = fitness
        print('Generation %d best fitness = %f' % (i, bestfitness))

        if fitness[0] > bestfitness:
            bestfitness = fitness[0]
            counter = 0
        else:
            counter += 1
        fitness_time[i] = time() - start
        if counter == 20 and stop_condition:
            fitness_hist[i:, :] = fitness[0]
            break
    best_sol = pop[0]
    if create_model:
        model = ProbabilisticModel('umd')
        print('build model input shape: ',
              Chromosome.genes_to_numpy(pop).shape)
        model.buildModel(Chromosome.genes_to_numpy(pop))
        src_models.append(model)

    return src_models, best_sol, fitness_hist, fitness_time


def transfer_bandit(problem,
                    dims,
                    reps,
                    trans,
                    psize=50,
                    gen=100,
                    src_models=[],
                    time_limits=None,
                    sample_size=None):

    if time_limits is not None:
        assert len(
            time_limits
        ) == reps, "time_limits length does not match the repetition numbers"
    else:
        time_limits = [float('inf')] * reps

    if sample_size is None:
        sample_size = psize

    if trans['transfer'] and (not src_models):
        raise ValueError(
            'No probabilistic models stored for transfer optimization.')

    init_func = lambda n: np.round(np.random.rand(n))
    model_num = len(src_models)
    fitness_hist = np.zeros([reps, gen, psize])
    fitness_time = np.zeros((
        reps,
        gen,
    ))
    alpha = list()
    prob = list()
    avg_runtime = 0
    time_passed = 0
    for rep in range(reps):
        print('------------------------ rep: {} ---------------------'.format(
            rep))
        start = time()
        alpha_rep = []

        pop = get_pop_init(psize, dims, init_func)
        for i in range(psize):
            pop[i].fitness_calc(problem)

        bestfitness = np.max(pop).fitness
        fitness = Chromosome.fitness_to_numpy(pop)

        fitness_hist[rep, 0, :] = fitness

        prob_rep = np.zeros((gen, model_num))
        prob_rep[0, :] = (1 / model_num) * np.ones(
            model_num)  # Initial uniform probablity of src model selection
        cum_rew = np.zeros((model_num))  # Initial source rewards

        fitness_time[rep, 0] = time() - start
        time_passed = fitness_time[rep, 0]
        print('Generation 0 best fitness = %f' % bestfitness)
        for i in range(1, gen):
            start = time()
            cfitness = np.zeros(psize)
            if trans['transfer'] and i % trans['delta'] == 0:
                # Selecting the the probability model
                idx = roulette_wheel_selection(
                    prob_rep[i - 1, :]
                )  # Selecting a model using roulette wheel selection technique
                
                sel_model = [src_models[idx]]

                # Applying EM algorithm and sampling from the mixture model
                mixModel = MixtureModel(sel_model)
                mixModel.createTable(Chromosome.genes_to_numpy(pop), True,
                                     'umd')
                
                mixModel.EMstacking()
                alpha_rep.append(mixModel.alpha)
                
                mixModel.mutate(version='bandit')
                offsprings_tmp = mixModel.sample(sample_size)

                # Calculating Fitness
                offsprings = np.array([
                    Chromosome(offspring_tmp)
                    for offspring_tmp in offsprings_tmp
                ])

                for j in range(sample_size):
                    cfitness[j] = offsprings[j].fitness_calc(problem)

                # Getting reward using importance sampling
                rew = mixModel.reward(model_num, offsprings_tmp, cfitness)

                # Updating probablities and rewards using exp3 algorithm
                prob_rep[i, :], cum_rew = EXP3(model_num, rew, idx, cum_rew,
                                               prob_rep[i - 1])

                #################################################################
                # print('Probabilities: {}'.format(prob_rep[i,:]))
                print('Mixture coefficients: %s' % np.array(mixModel.alpha))
            else:
                # Crossover & Mutation
                offsprings = total_crossover(pop)
                for j in range(psize):
                    offsprings[j].mutation(1 / dims)

                # Fitness Calculation
                for j in range(psize):
                    cfitness[j] = offsprings[j].fitness_calc(problem)

                prob_rep[i, :] = prob_rep[i - 1, :]
                # print('prob_rep[i,:] ', prob_rep[i,:])

            # Selection
            pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                           np.concatenate((fitness, cfitness)),
                                           psize)

            bestfitness = fitness[0]
            fitness_hist[rep, i, :] = fitness
            fitness_time[rep, i] = time() - start
            time_passed += fitness_time[rep, i]
            print('Generation %d best fitness = %f' % (i, bestfitness))
            if time_limits[rep] < time_passed:
                break

        alpha.append(alpha_rep)
        prob.append(prob_rep)
    return fitness_hist, alpha, prob, fitness_time

def get_args():
    pass

def check_args(args):
    pass


def main(args=False):

    ################# Preconfigurations ##############
    if args is False:
        args = get_args()

    check_args(args)
    models_path = 'models'
    source_models_path = os.path.join(models_path, 'knapsack_source_models')
    knapsack_problem_path = 'problems/knapsack'
    src_models = None
    gen = args.gen
    psize = args.psize

    src_models, target_problem = source_generator(args.src_version,
                                                  knapsack_problem_path,
                                                  source_models_path,
                                                  args.buildmodel,
                                                  args.stop_condition)

    reps = args.reps

    trans = {}
    trans['transfer'] = args.transfer
    trans['delta'] = args.delta

    return transfer_bandit(target_problem,
                           1000,
                           reps,
                           trans,
                           psize=psize,
                           gen=gen,
                           src_models=src_models,
                           time_limits=args.time_limits,
                           sample_size=args.sample_size)


if __name__ == '__main__':
    main()
