import sys
import numpy as np

from time import time
from evolution.individual import *
from to.probabilistic_model import ProbabilisticModel
from to.mixture_model import MixtureModel

from copy import deepcopy


def softmax(v):
    v_tmp = np.exp(v - np.max(v))
    v_tmp /= np.sum(v_tmp)
    return v_tmp

def norm_softmax(v, intensity=100):
    v_min = np.min(v)
    if v_min < 0:
        new_v = v[:] + np.abs(v_min)
    else:
        new_v = v
    norm = (new_v / np.max(new_v)) * intensity
    return softmax(norm)


class Chromosome(Individual):
    def __init__(self, n, init_func=np.random.rand):
        super().__init__(n, init_func=init_func)

    def mutation(self, mprob):
        mask = np.random.rand(self.genes.shape[0]) < mprob
        self.genes[mask] = np.abs(1 - self.genes[mask])

    def fitness_calc(self, problem):  
        # You can implement this in a more optmized way using vectorizatioin but it will hurt modularity
        if type(problem) == str:
            return self._fitness_unitation_calc(problem)

        if self.fitness >= 0:
            return self.fitness
        weights = problem['w']
        profits = problem['p']
        ratios = profits / weights
        selected_items = self.genes == 1
        total_weight = np.sum(weights[selected_items])
        total_profit = np.sum(profits[selected_items])

        if total_weight > problem['cap']:  # Repair solution
            selections = np.sum(selected_items)
            r_index = np.zeros((selections, 2))
            counter = 0

            for j in range(len(self.genes)):
                if selected_items[j] == 1:
                    r_index[counter, 0] = ratios[j]
                    r_index[counter, 1] = int(j)
                    counter = counter + 1
                if counter >= selections:
                    break

            r_index = r_index[r_index[:, 0].argsort()[::-1]]
            counter = selections - 1
            while total_weight > problem['cap']:
                l = int(r_index[counter, 1])
                selected_items[l] = 0
                total_weight = total_weight - weights[l]
                total_profit = total_profit - profits[l]
                counter = counter - 1

        self.fitness = total_profit
        return self.fitness

    def _fitness_unitation_calc(self, problem):  # You can implement this in a more optmized way using vectorizatioin but it will hurt modularity
        
        if self.fitness >= 0:
            return self.fitness
        if problem == 'onemin':
            self.fitness = len(self.genes) - np.sum(self.genes)
        elif problem == 'onemax':
            self.fitness = np.sum(self.genes)
        elif problem == 'trap5':
            self.fitness = 0
            for i in range(0, len(self.genes), 5):
                ones_num = np.sum(self.genes[i:i + 5])
                if ones_num == 5:
                    self.fitness += 5
                else:
                    self.fitness += 4 - ones_num
        elif problem == '3deceptive':
            self.fitness = 0
            for i in range(0, len(self.genes), 3):
                ones_num = np.sum(self.genes[i:i + 3])
                if ones_num == 0:
                    self.fitness += 0.8
                elif ones_num == 1:
                    self.fitness += 0.9
                elif ones_num == 2:
                    self.fitness += 0
                else:
                    self.fitness += 1
        elif problem == '6bipolar':
            self.fitness = 0
            for i in range(0, len(self.genes), 6):
                ones_num = np.sum(self.genes[i:i + 6])
                if ones_num == 0 or ones_num == 6:
                    self.fitness += 1
                elif ones_num == 1 or ones_num == 5:
                    self.fitness += 0
                elif ones_num == 2 or ones_num == 4:
                    self.fitness += 0.8
                else:
                    self.fitness += 0.9
        else:
            raise ValueError('{} problem has not defined'.format(problem))

        return self.fitness


class StrategyChromosome(Individual):
    def __init__(self, n, init_func=np.random.rand):
        super().__init__(n, init_func=init_func)

    def mutation_enhanced(self,
                          mute_strength,
                          genes_min_bounds=None,
                          genes_max_bounds=None,
                          genes_num=None,
                          lr=0.9):
        self.genes = (1 - lr) * self.genes + lr * norm_softmax(mute_strength)
        
        if np.sum(self.genes) == 0:
            self.genes[-1] = 1

    def fitness_calc_enhanced(self,
                              problem,
                              src_models,
                              target_model,
                              sample_size,
                              mutation_strength,
                              samples_count,
                              max_sampling_num=None,
                              mutation=True,
                              problem_type='knapsack'):
        start = time()

        if (not all(self.genes == self.genes[0])):
            termination_mask = self.genes > (1 / (len(src_models) + 1) *
                                             0.01) * 1.0  # BUG
            genes = termination_mask * self.genes

            genes = genes / np.sum(genes)

        else:
            genes = self.genes
            print('first iteration!? not neutralized')

        # Initializing the weights of the mixture model with
        mixModel = MixtureModel(src_models, alpha=genes)
        mixModel.add_target_model(target_model)

        offsprings, mutation_strength, samples_count, fitness_mean = \
            mixModel.sample_enhanced(sample_size, problem, mutation_strength,
                                    samples_count, max_sampling_num, mutation=mutation,
                                    problem_type=problem_type)

        self.fitness = fitness_mean

        self.fitness_calc_time = time() - start

        return self.fitness, offsprings, mutation_strength, samples_count

    def fitness_calc_pole(self,
                          net,
                          cart,
                          s_len,
                          src_models,
                          target_model,
                          sample_size,
                          mutation_strength,
                          samples_count,
                          solution_found=None):

        start = time()

        if not all(self.genes == self.genes[0]):
            termination_mask = self.genes > (1 / (len(src_models) + 1) *
                                             0.01) * 1.0
            genes = termination_mask * self.genes

            genes = genes / np.sum(genes)
            print('genes after normalization: ', genes)
        else:
            genes = self.genes
            print('first iteration!? not neutralized')

        # Initializing the weights of the mixture model with
        mixModel = MixtureModel(src_models, alpha=genes)
        mixModel.add_target_model(target_model)

        offsprings, mutation_strength, samples_count, fitness_mean, eval_num = \
            mixModel.sample_enhanced(sample_size, cart, mutation_strength,
                                    samples_count, net=net, s_len=s_len, mutation=False,
                                    solution_found=solution_found, problem_type='pole')

        self.fitness = fitness_mean

        self.fitness_calc_time = time() - start
        print('self.fitness_calc_time (m): ', self.fitness_calc_time / 60)
        # best_offspring = np.max(offsprings)
        return self.fitness, offsprings, mutation_strength, samples_count, eval_num


class ChromosomePole(Individual):
    def __init__(self, n, init_func=np.random.rand):
        super().__init__(n, init_func=init_func)

    def mutation(self, mum, dims):
        child = self.genes.copy()

        for i in range(dims):
            if np.random.rand() < 1 / dims:
                u = np.random.rand()
                if u <= 0.5:
                    delta = (2 * u)**(1 / (1 + mum)) - 1
                    child[i] = self.genes[i] + delta * self.genes[i]
                else:
                    delta = 1 - (2 * (1 - u))**(1 / (1 + mum))
                    child[i] = self.genes[i] + delta * (1 - self.genes[i])
        self.genes = child

    def fitness_calc(
            self, net, cart, sLen
    ):  
        # You can implement this in a more optmized way using vectorizatioin but it will hurt modularity
        if self.fitness >= 0:
            return self.fitness
        net.init_weight(self.genes)
        self.fitness = net.evaluate(cart, sLen)
        return self.fitness


class ChromosomeKA(ChromosomePole):
    
    def fitness_calc(
            self, arm, max_angle, target_pos
    ):  
        # You can implement this in a more optmized way using vectorizatioin but it will hurt modularity
        if self.fitness >= 0:
            return self.fitness

        angles = self.genes
        neg_angles_ind = angles < (0.0)
        one_pos_angles_ind = angles > (1.0)
        if any(np.logical_or(neg_angles_ind, one_pos_angles_ind)):
            return ((np.sum(np.abs(angles[neg_angles_ind])) +
                     np.sum(angles[one_pos_angles_ind] - 1)) * -1) + -3

        command = (angles - 0.5) * max_angle

        ef = arm.fw_kinematics(command)

        self.fitness = -np.linalg.norm(ef - target_pos)
        return self.fitness
