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
from utils.kinematic_arm import get_arm, get_distance_func
from utils.data_manipulators import *

def transfer_cc(problem,
                src_models,
                n_vars,
                psize=100,
                sample_size=100,
                gen=100,
                muc=10,
                mum=10,
                initial_genes_value=0,
                initial_lr=.9,
                reps=1,
                delta=2,
                build_model=False):

    if not src_models:
        raise ValueError(
            'No probabilistic models stored for transfer optimization.')

    fitness_hist = np.zeros([reps, gen, psize])
    fitness_time = np.zeros((reps, gen,))

    genes_list = list()

    dims_s2 = len(src_models)+1
    init_func_es = lambda n: np.ones(n)*initial_genes_value

    init_func = lambda n: np.random.rand(n)

    pop = None
    target_avg_fitness = 0
    for rep in range(reps):
        print('------------------------ rep: {} ---------------------'.format(rep))
        start = time()
        alpha_rep = []
        pop = get_pop_init(psize, n_vars, init_func, p_type='arm')

        second_specie = StrategyChromosome(dims_s2, init_func=init_func_es)
        mutation_strength = np.zeros(dims_s2) -2*np.sqrt(2)
        samples_count = np.zeros(dims_s2)
        lr = initial_lr
        genes_hist = []

        cfitness = np.zeros(psize)
        for j in range(psize):
            cfitness[j] = pop[j].fitness_calc(*problem)

        bestfitness = np.max(pop).fitness
        fitness = Chromosome.fitness_to_numpy(pop)
        fitness_hist[rep, 0, :] = fitness
        fitness_time[rep, 0] = time() - start
        print('Generation 0 best fitness = %f' % bestfitness)
        for i in range(1, gen):
            start = time()

            if i % delta == 0:
                target_array = Chromosome.genes_to_numpy(pop)
                shared_target_fitness = np.mean(fitness)

                second_specie_offspring = deepcopy(second_specie)

                if i // delta != 1:
                    mutation_strength[-1] = shared_target_fitness
                    second_specie_offspring. \
                      mutation_enhanced(mutation_strength, 0, 1, lr=lr)
                print('mutation_strength: ', mutation_strength)
                target_model = ProbabilisticModel(modelType='mvarnorm')

                target_model.buildModel(target_array)

                _, offsprings, mutation_strength, samples_count = second_specie_offspring . \
                                fitness_calc_enhanced(problem, src_models,
                                                target_model, sample_size, mutation_strength,
                                                samples_count, problem_type='arm')


                cfitness = Chromosome.fitness_to_numpy(offsprings)
                
                if second_specie.fitness <= second_specie_offspring.fitness:
                    second_specie = second_specie_offspring
                
                print(second_specie.genes)
                genes_hist.append(second_specie.genes)
                
                #################################################################
                # print('Probabilities: {}'.format(prob_rep[i,:]))
                # print('Genese: %s' % np.array(second_specie_offspring.genes))           
            else:
                # Crossover & Mutation
                randlist = np.random.permutation(psize)
                offsprings = np.ndarray(psize, dtype=object)
                for j in range(0, psize, 2):
                    offsprings[j] = ChromosomeKA(n_vars)
                    offsprings[j + 1] = ChromosomeKA(n_vars)
                    p1 = randlist[j]
                    p2 = randlist[j + 1]
                    offsprings[j].genes, offsprings[j + 1].genes = sbx_crossover(
                                                        pop[p1], pop[p2], muc, n_vars)
                    offsprings[j].mutation(mum, n_vars)
                    offsprings[j + 1].mutation(mum, n_vars)
                     
                # Fitness Calculation
                cfitness = np.zeros(psize)
                for j in range(psize):
                    cfitness[j] = offsprings[j].fitness_calc(*problem)
                

            # Selection
            pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                           np.concatenate((fitness, cfitness)),
                                           psize)

            fitness_hist[rep, i, :] = fitness
            fitness_time[rep, i] = time() - start

            if fitness[0] > bestfitness:
                bestfitness = fitness[0]

            print('Generation %d best fitness = %f' % (i, bestfitness))

        genes_list.append(genes_hist)

    return fitness_hist, genes_list, fitness_time

def get_args():
    pass

def check_args(args):
    pass


def main(args):

    if hasattr(args, 'src_models'):
        src_models = args.src_models
    else:
        # Loading Source Models
        src_models = Tools.load_from_file(args.source_models_path)
        print('---------------------- source models loaded---------------------')


    angular_range = args.max_angle * np.pi * 2.

    n_vars = args.joint_num
    arm = get_arm(args.target_length, n_vars)
    problem = (arm, angular_range, args.target_pos)

    return transfer_cc(problem, src_models, n_vars, sample_size=args.sample_size, psize=args.psize, gen=args.gen,
                  delta=args.delta, muc=10, mum=10, reps=args.reps, initial_genes_value=args.initial_genes_value,
                  initial_lr=args.initial_lr, build_model=args.buildmodel)


if __name__ == '__main__':
    args = get_args()
    main(args)
