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
from utils.ea import evolutionary_algorithm

def transfer_ea(problem,
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
    fitness_hist = np.zeros([reps, gen, psize])
    fitness_time = np.zeros((
        reps,
        gen,
    ))
    alpha = list()

    time_passed = 0

    for rep in range(reps):
        print('-------------------- rep: {} -------------------'.format(rep))
        alpha_rep = []
        start = time()
        pop = get_pop_init(psize, dims, init_func)
        for i in range(psize):
            pop[i].fitness_calc(problem)

        bestfitness = np.max(pop).fitness
        fitness = Chromosome.fitness_to_numpy(pop)
        fitness_hist[rep, 0, :] = fitness
        fitness_time[rep, 0] = time() - start
        time_passed = fitness_time[rep, 0]
        print('Generation 0 best fitness = %f' % bestfitness)
        for i in range(1, gen):
            start = time()
            if trans['transfer'] and i % trans['delta'] == 0:
                mixModel = MixtureModel(src_models)
                mixModel.createTable(Chromosome.genes_to_numpy(pop), True,
                                     'umd')
                mixModel.EMstacking()
                alpha_rep.append(mixModel.alpha)
                mixModel.mutate()
                offsprings = mixModel.sample(sample_size)

                offsprings = np.array(
                    [Chromosome(offspring) for offspring in offsprings])
                print('Mixture coefficients: %s' % np.array(mixModel.alpha))

            else:
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

            bestfitness = fitness[0]
            fitness_hist[rep, i, :] = fitness
            fitness_time[rep, i] = time() - start
            print('Generation %d best fitness = %f' % (i, bestfitness))
            time_passed += fitness_time[rep, i]
            if time_limits[rep] < time_passed:
                break
        alpha.append(alpha_rep)

    return fitness_hist, alpha, fitness_time

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
    # AMTEA solving KP_wc_ak
    reps = args.reps

    trans = {}
    trans['transfer'] = args.transfer
    trans['delta'] = args.delta
    if args.version == 'ea_time_scale':
        ea_fitness_hist = np.zeros((reps, gen, psize))
        ea_fitness_time = np.zeros((reps, gen))
        for i in range(reps):
            _, _, ea_fitness_hist[i, ...], ea_fitness_time[i, ...] = \
              evolutionary_algorithm(target_problem, 1000, src_models=src_models,
                                        gen=gen, psize=psize, stop_condition=False,
                                        create_model=False)
        return ea_fitness_hist, ea_fitness_time
    elif args.version == 'to':
        return transfer_ea(target_problem,
                           1000,
                           reps,
                           trans,
                           psize=psize,
                           gen=gen,
                           src_models=src_models,
                           time_limits=args.time_limits,
                           sample_size=args.sample_size)
    elif args.version == 'ea':
        return evolutionary_algorithm(target_problem,
                                      1000,
                                      src_models=src_models,
                                      stop_condition=args.stop_condition)
    else:
        raise ValueError('Version which you entered is not right')

if __name__ == '__main__':
    main()
