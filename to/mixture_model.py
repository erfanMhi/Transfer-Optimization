import numpy as np
from scipy.stats import multivariate_normal
from to.probabilistic_model import ProbabilisticModel
from evolution import chromosome as evolution
from copy import deepcopy



class MixtureModel(object):

    def __init__(self, allModels, alpha=False):
        self.model_list = allModels.copy()
        self.nModels = len(allModels)
        if alpha is False:
            self.alpha = (1/self.nModels)*np.ones(self.nModels)
        else:
            self.alpha = alpha

        self.probTable = None
        self.nSol = None
        self.__target_model_added = False

    def add_target_solutions(self, solutions, modelType):
        if not self.__target_model_added:
            self.nModels = self.nModels + 1
            self.model_list.append(ProbabilisticModel(modelType=modelType))
            self.model_list[-1].buildModel(solutions)
            self.__target_model_added = True
        else:
            raise Exception('Target model is already added.')

    def add_target_model(self, target_model):
        if not self.__target_model_added:
            self.nModels = self.nModels + 1
            self.model_list.append(target_model)
            self.target_model_added = True
        else:
            raise Exception('Target model is already added.')

    def createTable(self, solutions, CV, modelType, probs_RL=None):
        if CV:
            self.add_target_solutions(solutions, modelType)
            self.alpha = (1/self.nModels) * np.ones(self.nModels)
            nSol = solutions.shape[0]
            self.nSol = nSol
            self.probTable = np.ones([nSol, self.nModels])
 
            if probs_RL is None:
                for j in range(self.nModels-1):
                    self.probTable[:, j] = self.model_list[j].pdfEval(solutions) 
            else:
                for j in range(0, self.nModels-1):
                    self.probTable[:, j] = self.model_list[j].pdfEval(solutions) # Time complexity: O(pd)

            for i in range(nSol):  # Leave-one-out cross validation
                x = np.concatenate((solutions[:i, :], solutions[i+1:, :]))
                tModel = ProbabilisticModel(modelType=modelType)
                tModel.buildModel(x)
                self.probTable[i, -1] = tModel.pdfEval(solutions[[i], :])
        else:
            nSol = solutions.shape[0]
            self.probTable = np.ones([nSol, self.nModels])
            for j in range(self.nModels):
                self.probTable[:, j] = self.model_list[j].pdfEval(solutions)
            self.nSol = nSol

    def EMstacking(self, iterations=1):
        for t in range(iterations):
            print(t)
            talpha = self.alpha
            probVector = np.matmul(self.probTable, talpha.T)
            if any(probVector == 0):
                print('probVector: ', probVector)
                print('self.probTable: ', self.probTable)
                print('talpha: ', talpha)
            for i in range(self.nModels):
                talpha[i] = np.sum((1/self.nSol)*talpha[i]*self.probTable[:, i]/probVector)
            self.alpha = talpha

        if np.sum(np.isnan(self.alpha)) > 0:
            print('sanity check mutate')
            self.alpha = np.zeros(self.nModels)
            self.alpha[-1] = 1


    def mutate(self, version='normal'):
        modif_alpha = None
        
        modif_alpha = self.alpha + np.random.rand(self.nModels)*0.01

        total_alpha = np.sum(modif_alpha)
        if total_alpha == 0:
            self.alpha = np.zeros(self.nModels)
            self.alpha[-1] = 1
        else:
            self.alpha = modif_alpha/total_alpha

        # Sanity check
        if np.sum(np.isnan(self.alpha)) > 0:
            print('sanity check mutate')
            self.alpha = np.zeros(self.nModels)
            self.alpha[-1] = 1


    def sample(self, nSol, samplesRL=None, preprocess=False):
        
        if preprocess:
            i = 0
            while any(self.alpha[self.alpha!=0]<(1/nSol - np.finfo(np.float32).eps)):
                self.alpha[self.alpha<(1/nSol - np.finfo(np.float32).eps)] = 0
                self.alpha = self.alpha/np.sum(self.alpha)
                i += 1

        indSamples = np.ceil(nSol*self.alpha).astype(int)
        solutions = np.array([])
        for i in range(self.nModels):
            if indSamples[i] == 0:
                pass
            elif i == self.nModels - 2 and samplesRL is not None:
                solutions = np.vstack([solutions, samplesRL]) if solutions.size else samplesRL
            else:
                sols = self.model_list[i].sample(indSamples[i])
                solutions = np.vstack([solutions, sols]) if solutions.size else sols
        solutions = solutions[np.random.permutation(solutions.shape[0]), :]
        solutions = solutions[:nSol, :]
        return solutions


    def sample_enhanced(self, nSol, problem, mutation_strength, 
                        samples_count, max_sampling_num=None, solution_found=None,
                        problem_type='knapsack', net=None, s_len=None, mutation=True):
        """
        This sampling function only works for sTrEvo algorithm
        """

        if max_sampling_num is None:
            max_sampling_num = nSol
        indSamples = np.ceil(nSol*self.alpha).astype(int)

        solutions = []
        added_solutions = []
        solutions_idx = []
        for i in range(self.nModels):
            if indSamples[i] == 0:
                pass
            else:
                
                sampling_size = min(max_sampling_num, indSamples[i])
                sols_idx = np.ones(sampling_size) * i
                sols = self.model_list[i].sample(sampling_size)

                solutions = np.append(solutions, sols, axis=0) if len(solutions) else deepcopy(sols)
                solutions_idx = np.append(solutions_idx, sols_idx, axis=0) if len(sols_idx) else deepcopy(sols_idx)

        
        perm_indexes = np.random.permutation(len(solutions))
        solutions_num = min(nSol, len(solutions))
        solutions = solutions[perm_indexes][:solutions_num]
        solutions_idx = solutions_idx[perm_indexes][:solutions_num].astype(np.int)
        

        # Fitness Evaluation + Mutation_strength Update
        offsprings = []
        fitness_mean = 0
        
        func_eval_num = 0
        for solution, src_idx in zip(solutions, solutions_idx):
            
            if problem_type == 'knapsack':
                offsprings.append(evolution.Chromosome(solution))
                fitness = offsprings[-1].fitness_calc(problem)
            elif problem_type == 'pole':
                offsprings.append(evolution.ChromosomePole(solution))
                fitness = offsprings[-1].fitness_calc(net, problem, s_len)
                if not solution_found.value:
                    func_eval_num += 1
                if fitness - 2000 > -0.0001:
                    solution_found.value = True
            elif problem_type == 'arm':
                offsprings.append(evolution.ChromosomeKA(solution))
                fitness = offsprings[-1].fitness_calc(*problem)
            else:
                raise ValueError('Problem_type is wrong')
            
            fitness_mean += fitness
            if src_idx != self.nModels-1:
                samples_count[src_idx] += 1
                mutation_strength[src_idx] += (1/samples_count[src_idx])*(fitness - mutation_strength[src_idx])
        
        fitness_mean = fitness_mean/solutions_num


        # Sanity check
        if len(offsprings) != solutions_num:
            raise ValueError('offsprings length does not match the number of solutions')
        
        if solution_found is not None:     
            return offsprings, mutation_strength, samples_count, fitness_mean, func_eval_num
        else:
            return offsprings, mutation_strength, samples_count, fitness_mean

    def n_samples(self, ind, nSol):
        return np.ceil(nSol * self.alpha[ind]).astype(int)

