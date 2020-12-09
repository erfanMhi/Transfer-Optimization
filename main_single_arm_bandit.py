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

def transfer_bandit(problem,
                src_models,
                n_vars,
                psize=100,
                sample_size=100,
                gen=100,
                muc=10,
                mum=10,
                reps=1,
                delta=2,
                build_model=False):

    if not src_models:
        raise ValueError(
            'No probabilistic models stored for transfer optimization.')

    fitness_hist = np.zeros([reps, gen, psize])
    fitness_time = np.zeros((reps, gen,))
    alpha = list()

    init_func = lambda n: np.random.rand(n)
    alpha = list()
    prob = list()

    model_num = len(src_models)
    pop = None
    for rep in range(reps):
        print('------------------------ rep: {} ---------------------'.format(
            rep))
        start = time()
        alpha_rep = []

        pop = get_pop_init(psize, n_vars, init_func, p_type='arm')
        for i in range(psize):
            pop[i].fitness_calc(*problem)

        bestfitness = np.max(pop).fitness
        fitness = Chromosome.fitness_to_numpy(pop)

        fitness_hist[rep, 0, :] = fitness

        prob_rep = np.zeros((gen, model_num))
        prob_rep[0, :] = (1 / model_num) * np.ones(
            model_num)  # Initial uniform probablity of src model selection
        cum_rew = np.zeros((model_num))  # Initial source rewards

        fitness_time[rep, 0] = time() - start
        print('Generation 0 best fitness = %f' % bestfitness)
        for i in range(1, gen):
            start = time()
            if i % delta == 0:
                
                cfitness = np.zeros(sample_size)
                
                # Selecting the the probability model
                idx = roulette_wheel_selection(
                    prob_rep[i - 1, :]
                )  # Selecting a model using roulette wheel selection technique
                
                sel_model = [src_models[idx]]

                # Applying EM algorithm and sampling from the mixture model
                mixModel = MixtureModel(sel_model)
                mixModel.createTable(Chromosome.genes_to_numpy(pop), True,
                                     'mvarnorm')
                
                mixModel.EMstacking()
                
                alpha_rep.append(mixModel.alpha)
                
                mixModel.mutate(version='bandit')
                
                offsprings_tmp = mixModel.sample(sample_size)

                # Calculating Fitness
                offsprings = np.array([
                    ChromosomeKA(offspring_tmp)
                    for offspring_tmp in offsprings_tmp
                ])
                
                for j in range(sample_size):
                    cfitness[j] = offsprings[j].fitness_calc(*problem)
                
                # Getting reward using importance sampling
                rew = mixModel.reward(model_num, offsprings_tmp, cfitness)
                
                # Updating probablities and rewards using exp3 algorithm
                prob_rep[i, :], cum_rew = EXP3(model_num, rew, idx, cum_rew,
                                               prob_rep[i - 1])

                #################################################################
                # print('Mixture coefficients: %s' % np.array(mixModel.alpha))
            else: 
                # Crossover & Mutation
                randlist = np.random.permutation(psize)
                offsprings = np.ndarray(psize, dtype=object)
                for j in range(0, psize, 2):
                    offsprings[j] = ChromosomeKA(n_vars)
                    offsprings[j + 1] = ChromosomeKA(n_vars)
                    p1 = randlist[j]
                    p2 = randlist[j + 1]
                    offsprings[j].genes, offsprings[j +
                                                    1].genes = sbx_crossover(
                                                        pop[p1], pop[p2], muc,
                                                        n_vars)
                    offsprings[j].mutation(mum, n_vars)
                    offsprings[j + 1].mutation(mum, n_vars)

            
                # Fitness Calculation
                cfitness = np.zeros(psize)
                for j in range(psize):
                    cfitness[j] = offsprings[j].fitness_calc(*problem)

                prob_rep[i, :] = prob_rep[i - 1, :]
            
            if i % delta == 0:
                print('cfitness mean: ', np.mean(cfitness))

            # Selection
            pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                           np.concatenate((fitness, cfitness)),
                                           psize)

            fitness_hist[rep, i, :] = fitness
            fitness_time[rep, i] = time() - start

            if fitness[0] > bestfitness:
                bestfitness = fitness[0]

            print('Generation %d best fitness = %f' % (i, bestfitness))
        
        alpha.append(alpha_rep)
        prob.append(prob_rep)

    return fitness_hist, alpha, prob, fitness_time

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

    return transfer_bandit(problem, src_models, n_vars, psize=args.psize, sample_size=args.sample_size, gen=args.gen,
                  delta=args.delta, muc=10, mum=10, reps=args.reps, build_model=args.buildmodel)


if __name__ == '__main__':
    args = get_args()
    main(args)
