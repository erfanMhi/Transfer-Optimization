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
from utils.double_pole_physics import PoledCart
from utils.neural_network import Net

def evolutionary_algorithm(sLen,
                           psize=100,
                           gen=100,
                           muc=10,
                           mum=10,
                           stop_condition=True,
                           create_model=True):

    src_model = None

    fitness_hist = np.zeros((gen, psize))
    fitness_time = np.zeros((gen))

    cart = PoledCart(sLen)

    n_input = 6
    n_hidden = 10
    n_output = 1
    net = Net(n_input, n_hidden, n_output)
    n_vars = net.nVariables

    init_func = lambda n: 12 * np.random.rand(n) - 6
    pop = get_pop_init(psize, n_vars, init_func, p_type='double_pole')
    start = time()

    for j in range(psize):
        pop[j].fitness_calc(net, cart, sLen)

    bestfitness = np.max(pop).fitness
    fitness = Chromosome.fitness_to_numpy(pop)
    fitness_hist[0, :] = fitness

    fitness_time[0] = time() - start
    counter = 0  # Generation Repetition without fitness improvement counter
    for i in range(1, gen):
        start = time()
        randlist = np.random.permutation(psize)
        offsprings = np.ndarray(psize, dtype=object)

        # Crossover & Mutation
        for j in range(0, psize, 2):
            offsprings[j] = ChromosomePole(n_vars)
            offsprings[j + 1] = ChromosomePole(n_vars)
            p1 = randlist[j]
            p2 = randlist[j + 1]
            offsprings[j].genes, offsprings[j + 1].genes = sbx_crossover(
                pop[p1], pop[p2], muc, n_vars)
            offsprings[j].mutation(mum, n_vars)
            offsprings[j + 1].mutation(mum, n_vars)

        # Fitness Calculation
        cfitness = np.zeros(psize)
        for j in range(psize):
            # print(pop[j].genes)
            cfitness[j] = offsprings[j].fitness_calc(net, cart, sLen)

        # Selection
        pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                       np.concatenate((fitness, cfitness)),
                                       psize)

        fitness_hist[i, :] = fitness

        if fitness[0] > bestfitness:
            bestfitness = fitness[0]
            counter = 0
        else:
            counter += 1

        print('Generation %d best fitness = %f' % (i, bestfitness))
        if fitness[0] - 2000 > -0.0001 and stop_condition:
            print('Solution found!')
            fitness_hist[i:, :] = fitness[0]
            break

        fitness_time[i] = time() - start

    best_sol = pop[0]
    if create_model and fitness_hist[-1, 0] - 2000 > -0.0001:
        model = ProbabilisticModel('mvarnorm')
        print('build model input shape: ',
              Chromosome.genes_to_numpy(pop).shape)
        model.buildModel(Chromosome.genes_to_numpy(pop))
        print("Model built successfully!")
        src_model = model
    elif not create_model:
        print("Evolutionary algorithm didn't reach the criteria!")
        # src_models.append(model)

    return src_model, best_sol, fitness_hist, fitness_time


class IsFound:
    value = None


def transfer_cc(sLen,
                src_models,
                psize=100,
                gen=100,
                muc=10,
                mum=10,
                reps=1,
                delta=2,
                initial_genes_value=1/14,
                initial_lr=0.9,
                build_model=True):

    s_len = sLen
    if not src_models:
        raise ValueError(
            'No probabilistic models stored for transfer optimization.')

    init_func = lambda n: 12 * np.random.rand(n) - 6

    fitness_hist = np.zeros([reps, gen, psize])
    fitness_time = np.zeros((
        reps,
        gen,
    ))
    genes_list = list()

    dims_s2 = len(src_models) + 1
    init_func_es = lambda n: np.ones(n) * initial_genes_value

    cart = PoledCart(sLen)

    n_input = 6
    n_hidden = 10
    n_output = 1
    net = Net(n_input, n_hidden, n_output)
    n_vars = net.nVariables

    model_num = len(src_models)

    shared_target_fitness = None
    target_array = None

    pop = None
    func_eval_nums = []
    for rep in range(reps):
        print('-------------------- rep: {} -------------------'.format(rep))
        genes_hist = []

        start = time()
        # Evolution Strategy Initialization
        second_specie = StrategyChromosome(dims_s2, init_func=init_func_es)
        second_specie.fitness = 0
        mutation_strength = np.zeros(dims_s2)
        samples_count = np.zeros(dims_s2)
        lr = initial_lr

        func_eval_num = 0
        solution_found = IsFound()
        solution_found.value = False

        pop = get_pop_init(psize, n_vars, init_func, p_type='double_pole')
        for j in range(psize):
            pop[j].fitness_calc(net, cart, sLen)
            if not solution_found.value:
                func_eval_num += 1
            if pop[j].fitness - 2000 > -0.0001:
                solution_found.value = True
        bestfitness = np.max(pop).fitness
        fitness = Chromosome.fitness_to_numpy(pop)
        fitness_hist[rep, 0, :] = fitness
        fitness_time[rep, 0] = time() - start
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

                target_model = ProbabilisticModel(modelType='mvarnorm')

                target_model.buildModel(target_array)

                _, offsprings, mutation_strength, samples_count, eval_num = second_specie_offspring. \
                          fitness_calc_pole(net, cart, s_len, src_models,
                                          target_model, psize, mutation_strength,
                                          samples_count, solution_found=solution_found)

                func_eval_num += eval_num

                cfitness = Chromosome.fitness_to_numpy(offsprings)

                if second_specie.fitness <= second_specie_offspring.fitness:
                    second_specie = second_specie_offspring
                
                genes_hist.append(second_specie.genes)
                #################################################################
                # print('Probabilities: {}'.format(prob_rep[i,:]))
                # print('Genese: %s' % np.array(second_specie_offspring.genes))
            else:
                # Crossover & Mutation
                randlist = np.random.permutation(psize)
                offsprings = np.ndarray(psize, dtype=object)
                for j in range(0, psize, 2):
                    offsprings[j] = ChromosomePole(n_vars)
                    offsprings[j + 1] = ChromosomePole(n_vars)
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
                    cfitness[j] = offsprings[j].fitness_calc(net, cart, sLen)
                    if not solution_found.value:
                        func_eval_num += 1
                    if cfitness[j] - 2000 > -0.0001:
                        # func_eval_num = (i*psize + j+1)
                        solution_found.value = True

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
            
            if fitness[0] - 2000 > -0.0001 and build_model:
                print('Solution found!')
                fitness_hist[rep, i:, :] = fitness[0]
                break

        func_eval_nums.append(func_eval_num if solution_found.value else None)
        genes_list.append(genes_hist)

    return fitness_hist, genes_list, fitness_time, func_eval_nums


def transfer_ea(sLen,
                src_models,
                psize=100,
                gen=100,
                muc=10,
                mum=10,
                reps=1,
                delta=2,
                build_model=True):

    if not src_models:
        raise ValueError(
            'No probabilistic models stored for transfer optimization.')

    init_func = lambda n: 12 * np.random.rand(n) - 6

    fitness_hist = np.zeros([reps, gen, psize])
    fitness_time = np.zeros((
        reps,
        gen,
    ))
    alpha = list()

    cart = PoledCart(sLen)

    n_input = 6
    n_hidden = 10
    n_output = 1
    net = Net(n_input, n_hidden, n_output)
    n_vars = net.nVariables

    pop = None
    func_eval_nums = []
    sols_found = []
    for rep in range(reps):
        alpha_rep = []
        func_eval_num = 0
        solution_found = False
        pop = get_pop_init(psize, n_vars, init_func, p_type='double_pole')
        start = time()
        for j in range(psize):
            pop[j].fitness_calc(net, cart, sLen)
            if not solution_found:
                func_eval_num += 1
            if pop[j].fitness - 2000 > -0.0001:
                # func_eval_num = (i*psize + j+1)
                solution_found = True
        bestfitness = np.max(pop).fitness
        fitness = Chromosome.fitness_to_numpy(pop)
        fitness_hist[rep, 0, :] = fitness
        fitness_time[rep, 0] = time() - start
        print('Generation 0 best fitness = %f' % bestfitness)

        for i in range(1, gen):
            start = time()
            if i % delta == 0:
                mixModel = MixtureModel(src_models)
                mixModel.createTable(Chromosome.genes_to_numpy(pop), True,
                                     'mvarnorm')
                mixModel.EMstacking()
                mixModel.mutate()
                offsprings = mixModel.sample(psize)
                offsprings = np.array(
                    [ChromosomePole(offspring) for offspring in offsprings])
                alpha_rep = np.concatenate((alpha_rep, mixModel.alpha), axis=0)
                print('Mixture coefficients: %s' % np.array(mixModel.alpha))
            else:
                # Crossover & Mutation
                randlist = np.random.permutation(psize)
                offsprings = np.ndarray(psize, dtype=object)
                for j in range(0, psize, 2):
                    offsprings[j] = ChromosomePole(n_vars)
                    offsprings[j + 1] = ChromosomePole(n_vars)
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
                cfitness[j] = offsprings[j].fitness_calc(net, cart, sLen)
                if not solution_found:
                    func_eval_num += 1
                if cfitness[j] - 2000 > -0.0001:
                    # func_eval_num = (i*psize + j+1)
                    solution_found = True

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
            print(fitness[0])
            if fitness[0] - 2000 > -0.0001:
                print('Solution found!')
                fitness_hist[rep, i:, :] = fitness[0]
                func_eval_nums.append(func_eval_num)
                sols_found.append(solution_found)
                break

        print()

        alpha.append(alpha_rep)

    model = None
    print('fitness_hist: ', fitness_hist[0, -1, 0])
    if build_model and fitness_hist[0, -1, 0] - 2000 > -0.0001:
        model = ProbabilisticModel('mvarnorm')
        print('build model input shape: ',
              Chromosome.genes_to_numpy(pop).shape)
        model.buildModel(Chromosome.genes_to_numpy(pop))
        print("Model built successfully!")
        # src_model = model
    else:
        print("Evolutionary algorithm didn't reach the criteria!")

    if build_model:
        return fitness_hist[0, ...], alpha, fitness_time[0, ...], model
    else:
        return fitness_hist, alpha, fitness_time, sols_found, func_eval_nums


def get_args():
    pass

def check_args(args):
    if args.sample_size < args.sub_sample_size:
        raise ValueError('sub_sample_size has greater value than sample_size')

def main_source(s_poles_length,
                target_pole_len,
                gen=100,
                reps=50,
                src_save_dir='models/pole_models/src_model'):
    # src_models = np.ndarray(len(s_poles_length), dtype=object)
    src_models = []
    src_model = None
    if os.path.isfile(src_save_dir + '_{}.pkl'.format(s_poles_length[0])):
        src_model = Tools.load_from_file(src_save_dir +
                                         '_{}'.format(s_poles_length[0]))
        print(
            '---------------------- source model loaded (length: {}) ---------------------'
            .format(s_poles_length[0]))
    else:
        src_model, _, _, _ = evolutionary_algorithm(s_poles_length[0],
                                                    psize=100,
                                                    gen=gen,
                                                    muc=10,
                                                    mum=10,
                                                    stop_condition=True,
                                                    create_model=True)
        Tools.save_to_file(src_save_dir + '_{}'.format(s_poles_length[0]),
                           src_model)
        print(
            '---------------------- source model created (length: {}) ---------------------'
            .format(s_poles_length[0]))
    src_models.append(src_model)
    for i, s_len in enumerate(s_poles_length[1:]):
        if os.path.isfile(src_save_dir + '_{}.pkl'.format(s_len)):
            src_model = Tools.load_from_file(src_save_dir +
                                             '_{}'.format(s_len))
            src_models.append(src_model)
            print(
                '---------------------- source model loaded (length: {}) ---------------------'
                .format(s_len))
        else:
            while (True):
                print('-------------- S_Length: {} ------------'.format(s_len))
                _, _, _, src_model = transfer_ea(s_len,
                                                 src_models,
                                                 psize=100,
                                                 gen=gen,
                                                 muc=10,
                                                 mum=10,
                                                 reps=1,
                                                 build_model=True)

                if src_model is not None:
                    Tools.save_to_file(src_save_dir + '_{}'.format(s_len),
                                       src_model)
                    src_models.append(src_model)
                    print(
                        '---------------------- source model created (length: {}) ---------------------'
                        .format(s_len))
                    break

    fitness_hist, genes_list, fitness_time, func_eval_nums =  \
          transfer_cc(target_pole_len, src_models, psize=50, gen=100,
                  delta=2, muc=10, mum=10, reps=reps, build_model=False, initial_genes_value=1/14)

    Tools.save_to_file(
        'single_cc_pole_outcome',
        [fitness_hist, genes_list, fitness_time, func_eval_nums])
    solved_indices = np.array(func_eval_nums) != None
    solved_instances = np.array(func_eval_nums)[solved_indices]
    print('Function Evaluations: {}'.format(np.sum(solved_instances)))
    print('Solutions found: {}/{}'.format(np.sum(solved_indices), reps))


if __name__ == '__main__':
    src_poles_length = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.675, 0.7, 0.725, 0.75,
        0.775
    ]
    target_pole_len = 0.825
    main_source(src_poles_length, target_pole_len, reps=50, gen=100)
