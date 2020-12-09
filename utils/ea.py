
import concurrent.futures
import numpy as np
import argparse
import os

from copy import deepcopy
from time import time
from pprint import pprint
from evolution.operators import *
from to.probabilistic_model import ProbabilisticModel
from to.mixture_model import MixtureModel
from evolution.chromosome import *
from utils.double_pole_physics import PoledCart
from utils.neural_network import Net


def evolutionary_algorithm(problem, dims, psize=100, gen=100, src_models=[],
                           stop_condition=True, create_model=True, multi_proc=False,
                           workers=1):
    
  
  fitness_hist = np.zeros((gen, psize))
  fitness_time = np.zeros((gen))

  init_func = lambda n: np.round(np.random.rand(n))
  pop = get_pop_init(psize, dims, init_func)
  start = time()

  if multi_proc:
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        processes = []
        for j in range(psize):
          
            proc = executor.submit(pop[j].fitness_calc, problem)
            processes.append(proc)
          
        fitnesses = [proc.result() for proc in processes]

        for j in range(psize):
            pop[j].fitness = fitnesses[j]
  else:
    fitnesses = np.zeros(psize)
    for j in range(psize): 
      fitnesses[j] = pop[j].fitness_calc(problem)
  


  bestfitness = np.max(fitnesses)
  fitnesses = fitnesses
  fitness_hist[0, :] = fitnesses

  fitness_time[0] =  time() - start
  counter = 0 # Generation Repetition without fitness improvement counter
  for i in range(1, gen):
      start = time()

      # Crossover & Mutation
      offsprings = total_crossover(pop)
      for j in range(psize): offsprings[j].mutation(1/dims)
      
      # Fitness Calculation

      if multi_proc:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
          processes = []
          for j in range(psize):
            
              proc = executor.submit(offsprings[j].fitness_calc, problem)
              processes.append(proc)
            
          cfitnesses = [proc.result() for proc in processes]

          for j in range(psize):
              offsprings[j].fitness = cfitnesses[j]
      else:
        cfitnesses = np.zeros(psize)
        for j in range(psize): 
          cfitnesses[j] = offsprings[j].fitness_calc(problem)

      # Selection
      pop, fitnesses = total_selection(np.concatenate((pop, offsprings)),
                                 np.concatenate((fitnesses, cfitnesses)), psize)

      fitness_hist[i, :] = fitnesses
      print('Generation %d best fitness = %f' % (i, bestfitness))


      if fitnesses[0] > bestfitness:
          bestfitness = fitnesses[0]
          counter = 0
      else:
          counter += 1
      fitness_time[i] = time() - start
      if counter == 20 and stop_condition:
          fitness_hist[i:, :] = fitnesses[0]
          break

  best_sol = pop[0]
  if create_model:
    model = ProbabilisticModel('umd')
    model.buildModel(Chromosome.genes_to_numpy(pop))
    src_models.append(model)

  
  return src_models, best_sol, fitness_hist, fitness_time

def continues_ea(fitness_func, init_func, dim, psize=100, gen=100, muc=10,
                           mum=10, stop_condition=True, create_model=True):

    src_model = None

    fitness_hist = np.zeros((gen, psize))
    fitness_time = np.zeros((gen))
    
    class ChromosomeEA(ChromosomePole):
        def fitness_calc(self):
            if self.fitness != float('-inf'):
                return self.fitness
            self.fitness = fitness_func(self.genes)
            return self.fitness

    pop = get_pop_init(psize, dim, init_func, p_type=ChromosomeEA)
    start = time()

    for j in range(psize):
        pop[j].fitness_calc()


    bestfitness = np.max(pop).fitness
    fitness = Chromosome.fitness_to_numpy(pop)
    fitness_hist[0, :] = fitness

    fitness_time[0] =  time() - start
    print('Generation %d best fitness = %f' % (0, bestfitness))
    counter = 0 # Generation Repetition without fitness improvement counter
    for i in range(1, gen):
        start = time()
        randlist = np.random.permutation(psize)
        offsprings = np.ndarray(psize, dtype=object)

        # Crossover & Mutation
        for j in range(0, psize, 2):
            offsprings[j] = ChromosomeEA(dim)
            offsprings[j+1] = ChromosomeEA(dim)
            p1 = randlist[j]
            p2 = randlist[j+1]
            offsprings[j].genes, offsprings[j+1].genes = sbx_crossover(pop[p1], pop[p2], muc, dim)
            offsprings[j].mutation(mum, dim)
            offsprings[j+1].mutation(mum, dim)


        # Fitness Calculation
        cfitness = np.zeros(psize)
        for j in range(psize):
            cfitness[j] = offsprings[j].fitness_calc()

        # Selection
        pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                 np.concatenate((fitness, cfitness)), psize)

        fitness_hist[i, :] = fitness

        if fitness[0] > bestfitness:
            bestfitness = fitness[0]
            counter = 0
        else:
            counter += 1

        print('Generation %d best fitness = %f' % (i, bestfitness))

        fitness_time[i] = time() - start

    best_sol = pop[0]
    if create_model:
        model = ProbabilisticModel('mvarnorm')
        print('build model input shape: ', ChromosomePole.genes_to_numpy(pop).shape)
        model.buildModel(ChromosomePole.genes_to_numpy(pop))
        print('model mean: ', model.mean)
        print("Model built successfully!")
        src_model = model
    elif not create_model:
        print("Evolutionary algorithm didn't reach the criteria!")

    return src_model, best_sol, fitness_hist, fitness_time

def transfer_continues_ea(fitness_func, init_func, dim, src_models, psize=100, gen=100,
                 muc=10, mum=10, reps=1, delta=2, build_model=True):

    if not src_models:
        raise ValueError('No probabilistic models stored for transfer optimization.')
        
    class ChromosomeEA(ChromosomePole):
        def fitness_calc(self):
            if self.fitness != float('-inf'):
                return self.fitness
            self.fitness = fitness_func(self.genes)
            return self.fitness

    fitness_hist = np.zeros([reps, gen, psize])
    fitness_time = np.zeros((reps, gen,))
    alpha = list()

    pop = None
    for rep in range(reps):
        alpha_rep = []
        pop = get_pop_init(psize, dim, init_func, p_type=ChromosomeEA)
        start = time()
        for j in range(psize):
            pop[j].fitness_calc()
                
        bestfitness = np.max(pop).fitness
        fitness = Chromosome.fitness_to_numpy(pop)
        fitness_hist[rep, 0, :] = fitness
        fitness_time[rep, 0] = time() - start
        print('Generation 0 best fitness = %f' % bestfitness)

      
        for i in range(1, gen):
            start = time()
            if i % delta == 0:
                mixModel = MixtureModel(src_models)
                mixModel.createTable(Chromosome.genes_to_numpy(pop), True, 'mvarnorm')
                mixModel.EMstacking()
                mixModel.mutate()
                offsprings = mixModel.sample(psize)
                offsprings = np.array([ChromosomeEA(offspring) for offspring in offsprings])
                alpha_rep = np.concatenate((alpha_rep, mixModel.alpha), axis=0)
#                 print('Mixture coefficients: %s' % np.array(mixModel.alpha))
            else:
                # Crossover & Mutation
                randlist = np.random.permutation(psize)
                offsprings = np.ndarray(psize, dtype=object)
                for j in range(0, psize, 2):
                    offsprings[j] = ChromosomeEA(dim)
                    offsprings[j+1] = ChromosomeEA(dim)
                    p1 = randlist[j]
                    p2 = randlist[j+1]
                    offsprings[j].genes, offsprings[j+1].genes = sbx_crossover(pop[p1], pop[p2], muc, dim)
                    offsprings[j].mutation(mum, dim)
                    offsprings[j+1].mutation(mum, dim)
              

            
            # Fitness Calculation
            cfitness = np.zeros(psize)
            for j in range(psize): 
                cfitness[j] = offsprings[j].fitness_calc()

            if i % delta == 0:
                print('cfitness mean: ', np.mean(cfitness))
                print('cfitness max: ', np.max(cfitness))
                print('cfitness min: ', np.min(cfitness))


            # Selection
            pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                    np.concatenate((fitness, cfitness)), psize)


            fitness_hist[rep, i, :] = fitness
            fitness_time[rep, i] = time() - start

            if fitness[0] > bestfitness:
                bestfitness = fitness[0]


            print('Generation %d best fitness = %f' % (i, bestfitness))
            print('Generation %d mean fitness = %f' % (i, np.mean(fitness)))
      
        print()

        alpha.append(alpha_rep)
  
    model = None
    print('fitness_hist: ', fitness_hist[0, -1, 0])
    if build_model:
        model = ProbabilisticModel('mvarnorm')
        print('build model input shape: ', Chromosome.genes_to_numpy(pop).shape)
        model.buildModel(Chromosome.genes_to_numpy(pop))
        print('model mean: ', model.mean)
        print("Model built successfully!")
    else:
        print("Evolutionary algorithm didn't reach the criteria!")

    if build_model:
        return fitness_hist[0, ...], alpha, fitness_time[0, ...], model, np.max(pop).genes
    else:
        return fitness_hist, alpha, fitness_time


def transfer_ea(problem, dims, delta, psize=100, gen=100,
               create_model=True, stop_condition=True, 
               src_models=[]):
  # load probabilistic models

  if src_models is None or len(src_models) == 0:
      raise ValueError('No probabilistic models stored for transfer optimization.')

  init_func = lambda n: np.round(np.random.rand(n))
  fitness_hist = np.zeros([gen, psize])
  fitness_time = np.zeros((gen))

  alpha_rep = []
  counter = 0

  pop = get_pop_init(psize, dims, init_func)
  start = time()
  for i in range(psize): pop[i].fitness_calc(problem)

  bestfitness = np.max(pop).fitness
  fitness = Chromosome.fitness_to_numpy(pop)
  fitness_hist[0, :] = fitness
  fitness_time[0] = time() - start
  print('Generation 0 best fitness = %f' % bestfitness)
  for i in range(1, gen):

      start = time()
      if i % delta == 0:
          mixModel = MixtureModel(src_models) 
          mixModel.createTable(Chromosome.genes_to_numpy(pop), True, 'umd')
          mixModel.EMstacking()
          mixModel.mutate()
          offsprings = mixModel.sample(psize)

          offsprings = np.array([Chromosome(offspring) for offspring in offsprings])
          alpha_rep.append(mixModel.alpha)
          print('Mixture coefficients: %s' % np.array(mixModel.alpha))

          

      else:
          # Crossover & Mutation
          offsprings = total_crossover(pop)
          for j in range(psize): offsprings[j].mutation(1/dims)
            

        
      # Fitness Calculation
      cfitness = np.zeros(psize)
      for j in range(psize): 
        cfitness[j] = offsprings[j].fitness_calc(problem)

      # Selection
      pop, fitness = total_selection(np.concatenate((pop, offsprings)),
                                np.concatenate((fitness, cfitness)), psize)

      fitness_hist[i, :] = fitness


      if fitness[0] > bestfitness:
          bestfitness = fitness[0]
          counter = 0
      else:
          counter += 1

      fitness_time[i] = time() - start
      
      if counter == 20 and stop_condition:
          fitness_hist[i:, :] = fitness[0]
          break
        
      print('Generation %d best fitness = %f' % (i, bestfitness))

  best_sol = pop[0]
  src_model = None
  if create_model:
    src_model = ProbabilisticModel('umd')
    print('build model input shape: ', Chromosome.genes_to_numpy(pop).shape)
    src_model.buildModel(Chromosome.genes_to_numpy(pop))
    print('probOne_noisy: ', src_model.probOne_noisy)
    print('probZero_noisy: ', src_model.probZero_noisy)


  return src_model, best_sol, fitness_hist, fitness_time


def evolutionary_algorithm_pole(sLen, psize=100, gen=100, muc=10,
                             mum=10, stop_condition=True, create_model=True, 
                             multi_proc=False, workers=1):
    
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

  if multi_proc:
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
      processes = []
      for j in range(psize):
        
          proc = executor.submit(pop[j].fitness_calc, net, cart, sLen)
          processes.append(proc)
        
      fitnesses = [proc.result() for proc in processes]

      for j in range(psize):
          pop[j].fitness = fitnesses[j]
  else:
    fitnesses = np.zeros(psize)
    for j in range(psize): 
      fitnesses[j] = pop[j].fitness_calc(net, cart, sLen)

  
  bestfitness = np.max(fitnesses)
  fitness_hist[0, :] = fitnesses

  fitness_time[0] = start - time()
  counter = 0 # Generation Repetition without fitness improvement counter
  for i in range(1, gen):
      start = time()
      randlist = np.random.permutation(psize)
      offsprings = np.ndarray(psize, dtype=object)

      # Crossover & Mutation
      for j in range(0, psize, 2):
            offsprings[j] = ChromosomePole(n_vars)
            offsprings[j+1] = ChromosomePole(n_vars)
            p1 = randlist[j]
            p2 = randlist[j+1]
            offsprings[j].genes, offsprings[j+1].genes = sbx_crossover(pop[p1], pop[p2], muc, n_vars)
            offsprings[j].mutation(mum, n_vars)
            offsprings[j+1].mutation(mum, n_vars)

      
      # Fitness Calculation
      if multi_proc:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
          processes = []
          for j in range(psize):
            
              proc = executor.submit(offsprings[j].fitness_calc, net, cart, sLen)
              processes.append(proc)
            
          cfitnesses = [proc.result() for proc in processes]

          for j in range(psize):
              offsprings[j].fitness = cfitnesses[j]
      else:
        cfitnesses = np.zeros(psize)
        for j in range(psize): 
          cfitnesses[j] = offsprings[j].fitness_calc(net, cart, sLen)



      cfitnesses = np.zeros(psize)
      for j in range(psize):
          # print(pop[j].genes)
          cfitnesses[j] = offsprings[j].fitness_calc(net, cart, sLen)

      # Selection
      pop, fitnesses = total_selection(np.concatenate((pop, offsprings)),
                                 np.concatenate((fitnesses, cfitnesses)), psize)

      fitness_hist[i, :] = fitnesses

      if fitnesses[0] > bestfitness:
          bestfitness = fitnesses[0]
          counter = 0
      else:
          counter += 1

      print('Generation %d best fitness = %f' % (i, bestfitness))
      if fitnesses[0] - 2000 > -0.0001 and stop_condition:
          print('Solution found!')
          fitness_hist[i:, :] = fitnesses[0]
          break

      fitness_time[i] = time() - start

  best_sol = pop[0]
  if create_model and fitness_hist[-1, 0] - 2000 > -0.0001:
    model = ProbabilisticModel('mvarnorm')
    print('build model input shape: ', Chromosome.genes_to_numpy(pop).shape)
    model.buildModel(Chromosome.genes_to_numpy(pop))
    print("Model built successfully!")
    src_model = model
  elif not create_model:
    print("Evolutionary algorithm didn't reach the criteria!")
    # src_models.append(model)
  
  return src_model, best_sol, fitness_hist, fitness_time
