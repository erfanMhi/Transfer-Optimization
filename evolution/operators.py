import numpy as np
from abc import ABC, abstractmethod
from evolution.chromosome import Chromosome, ChromosomePole, ChromosomeKA
from to.probabilistic_model import ProbabilisticModel


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


def kid_generator(parent,
                  mute_strength,
                  genes_min_bounds=None,
                  genes_max_bounds=None,
                  genes_num=None):
    
    # no crossover, only mutation
    if genes_num == None:
        genes_num = len(parent)
    offspring = parent + mute_strength * np.random.randn(genes_num)
    offspring = np.clip(offspring, genes_min_bounds, genes_max_bounds)
    return offspring


def crossover(p1, p2, alpha=np.random.rand(), ctype='uniform'):
    if ctype == 'convex':
        return Chromosome(alpha * p1.genes + (1 - alpha) * p2.genes)
    # elif stype=='uniform':
    #   return max(np.random.choice(self.pop,k,False),key=lambda c:c.fitness)
    else:
        raise ValueError('Type of crossover which you entered is wrong')


def total_crossover(pop):
    dims = len(pop[0].genes)
    psize = len(pop)
    parent1 = pop[np.random.permutation(psize)]
    parent2 = pop[np.random.permutation(psize)]
    offsprings = np.ndarray(psize, dtype=object)
    tmp = np.random.rand(psize, dims)
    p1_selection = tmp >= 0.5
    p2_selection = tmp < 0.5
    for i in range(psize):
        offsprings[i] = Chromosome(dims)
        offsprings[i].genes[p1_selection[i]] = parent1[i].genes[
            p1_selection[i]]
        offsprings[i].genes[p2_selection[i]] = parent2[i].genes[
            p2_selection[i]]
    return offsprings


def roulette_wheel_selection(probs):
    mx = np.sum(probs)
    rnd = np.random.rand() * mx
    sm = 0
    for i, prob in enumerate(probs):
        sm += prob
        if rnd < sm:
            return i


def total_selection(pop, fitnesses, psize):
    index = np.argsort(-fitnesses)  # default argsort is ascending
    return pop[index[:psize]], fitnesses[index[:psize]]


def total_selection_pop(pop, psize):
    index = np.argsort(-pop)
    return pop[index[:psize]]


def selection(pop, stype='roulette', k=1):
    if stype == 'roulette':
        sum_fit = np.sum([ch.fitness for ch in pop])
        pick = np.random.uniform(0, sum_fit)
        current = 0
        for chromosome in pop:
            current += chromosome.fitness
            if current > pick:
                return chromosome
    elif stype == 'tournament':
        return max(np.random.choice(pop, k, False), key=lambda c: c.fitness)
    else:
        raise ValueError('Type of selection which you entered is wrong')


def get_pop_init(n, gn, init_func=np.random.rand, p_type='knapsack'):
    """[Initial Population Generator]
  
  Arguments:
      n {[int]} -- [Number of members of population]
      gn {[int]} -- [Number of individual's genes]
      init_func {[function]} -- the function which initialize each chromosome
  
  Returns:
      [np.ndarray] -- [Array of chromosomes]
  """
    if p_type == 'knapsack':
        return np.array([Chromosome(gn, init_func) for _ in range(n)])
    elif p_type == 'double_pole':
        return np.array([ChromosomePole(gn, init_func) for _ in range(n)])
    elif p_type == 'arm':
        return np.array([ChromosomeKA(gn, init_func) for _ in range(n)])
    elif type(p_type) == type(ChromosomePole):
        return np.array([p_type(gn, init_func) for _ in range(n)])


def selection_adoption(parent, offspring, mute_strength, genes_num=None):
    if genes_num == None:
        genes_num = len(parent)

    fp = parent.fitness
    fk = offspring.fitness
    p_target = 1 / 5
    if fp <= fk:  # kid better than parent
        parent = offspring
        ps = 1  # kid win -> ps = 1 (successful offspring)
    else:
        ps = 0
    mute_strength *= np.exp(1 / np.sqrt(genes_num + 1) * (ps - p_target))

    return parent, mute_strength, fp <= fk


def selection_adoption_v2(parent,
                          offspring,
                          mute_strength,
                          gen,
                          success_gen,
                          c=2,
                          genes_num=None):
    if genes_num == None:
        genes_num = len(parent)

    fp = parent.fitness
    fk = offspring.fitness
    p_target = 1 / 5
    if fp < fk:  # kid better than parent
        parent = offspring
        ps = 1.  # kid win -> ps = 1 (successful offspring)
    else:
        ps = 0.
    success_gen = success_gen + ps
    g = (success_gen) / gen
    c_sq = c**2
    if g > p_target:
        mute_strength = mute_strength / c_sq
    else:
        mute_strength = c_sq * mute_strength
    return parent, mute_strength, success_gen, fp < fk


def sbx_crossover(p1, p2, muc, dims):
    child1 = np.zeros(dims)
    child2 = np.zeros(dims)
    randlist = np.random.rand(dims)
    for i in range(dims):
        if randlist[i] <= 0.5:
            k = (2 * randlist[i])**(1 / (muc + 1))
        else:
            k = (1 / (2 * (1 - randlist[i])))**(1 / (muc + 1))
        child1[i] = 0.5 * (((1 + k) * p1.genes[i]) + (1 - k) * p2.genes[i])
        child2[i] = 0.5 * (((1 - k) * p1.genes[i]) + (1 + k) * p2.genes[i])
        if np.random.rand() < 0.5:
            tmp = child1[i]
            child1[i] = child2[i]
            child2[i] = tmp
    return child1, child2


def EXP3(n, reward, indx, C, prob):
    """
    Exponential weighting algorithm for multi-armed bandits
    Function returns:
    up_probb = Updated probability distribution over bandits
    up_C = updated cumulative reward vector
    Input arguments:
    n = number of bandits
    reward = Observed reward
    indx = Index of observed bandit
    C = cumulative reward vector from previous trial
    prob = Probability vector from previous trial
    T = Iteration number
    """
    gama = 0.1
    if len(C) == n or len(prob) == n:
        ValueError("Input vectors must have same length as number of bandits")

    up_C = C
    up_C[indx] = up_C[indx] + (reward / prob[indx])
    exp_C = np.exp(gama * up_C / n)
    exp_C[exp_C < 0] = 0
    up_prob = ((1 - gama) * exp_C / np.sum(exp_C)) + (gama / n)
    # print('up_prob ', up_prob)
    return up_prob, up_C

def create_unitation_optimum(global_optimum=True,
                             problem='onemin',
                             psize=100,
                             dims=120):
    pop = None
    if problem == 'onemin':
        pop = np.zeros((psize, dims))
    elif problem == 'onemax':
        pop = np.ones((psize, dims))
    elif problem == 'trap5':
        pop = np.ones((psize, dims))
        if not global_optimum:
            for i in range(psize):
                for j in range(0, dims, 5):
                    if np.random.rand() > 0.5:
                        pop[i, j:j + 5] = 0
    model = ProbabilisticModel('umd')
    model.buildModel(pop.mean(0))
