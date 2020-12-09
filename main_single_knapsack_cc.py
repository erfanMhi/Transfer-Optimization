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


def transfer_cc(problem,
                dims,
                reps,
                delta,
                psize=50,
                gen=100,
                src_models=[],
                time_limits=None,
                sample_size=None,
                initial_genes_value=0.2,
                initial_lr=0.9,
                max_sampling_num=None):

    if time_limits is not None:
        assert len(
            time_limits
        ) == reps, "time_limits length does not match the repetition numbers"
    else:
        time_limits = [float('inf')] * reps

    if sample_size is None:
        sample_size = psize

    if max_sampling_num is None:
        max_sampling_num = sample_size

    if not src_models:
        raise ValueError(
            'No probabilistic models stored for transfer optimization.')

    init_func = lambda n: np.round(np.random.rand(n))
    fitness_hist = np.zeros([reps, gen, psize])
    fitness_time = np.zeros((
        reps,
        gen,
    ))
    genes_list = list()

    dims_s2 = len(src_models) + 1
    init_func_es = lambda n: np.ones(n) * initial_genes_value

    shared_target_fitness = None
    target_array = None

    time_passed = 0

    for rep in range(reps):
        print('-------------------- rep: {} -------------------'.format(rep))

        start = time()
        genes_hist = []

        # Evolution Strategy Initialization
        second_specie = StrategyChromosome(dims_s2, init_func=init_func_es)
        second_specie.fitness = 0
        mutation_strength = np.zeros(dims_s2)
        samples_count = np.zeros(dims_s2)
        lr = initial_lr

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
            cfitness = np.zeros(psize)
            if i % delta == 0:
                target_array = Chromosome.genes_to_numpy(pop)
                shared_target_fitness = np.mean(fitness)

                second_specie_offspring = deepcopy(second_specie)

                if i // delta != 1:
                    mutation_strength[-1] = shared_target_fitness
                    second_specie_offspring. \
                      mutation_enhanced(mutation_strength, 0, 1, lr=lr)

                target_model = ProbabilisticModel(modelType='umd')

                target_model.buildModel(target_array)

                _, offsprings, mutation_strength, samples_count = second_specie_offspring. \
                            fitness_calc_enhanced(problem, src_models, target_model,
                                                sample_size, mutation_strength,
                                                samples_count, max_sampling_num)

                cfitness = Chromosome.fitness_to_numpy(offsprings)

                if second_specie.fitness <= second_specie_offspring.fitness:
                    second_specie = second_specie_offspring

                genes_hist.append(second_specie.genes)
                #################################################################
                # print('Probabilities: {}'.format(prob_rep[i,:]))
                # print('Genese: %s' % np.array(second_specie_offspring.genes))

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

        genes_list.append(genes_hist)

    return fitness_hist, genes_list, fitness_time

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

    #AMTEA solving KP_wc_ak
    reps = args.reps

    delta = args.delta
    return transfer_cc(target_problem,
                       1000,
                       reps,
                       delta,
                       psize=psize,
                       gen=gen,
                       src_models=src_models,
                       time_limits=args.time_limits,
                       sample_size=args.sample_size,
                       initial_lr=args.initial_lr,
                       initial_genes_value=args.initial_genes_value)


if __name__ == '__main__':
    main()
