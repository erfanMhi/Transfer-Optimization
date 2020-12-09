import pickle
import os
import numpy as np
from time import time
from utils.ea import evolutionary_algorithm

class Tools:
    @staticmethod
    def save_to_file(path, data):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(path):
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)

def source_generator(src_version, knapsack_problem_path, 
                    source_models_path, buildmodel, 
                    stop_condition):
    
  src_problems = []

  # Loading Problems Data
  if src_version == 'kp4-a':
    KP_sc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_ak'))
    KP_uc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_ak'))
    KP_wc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_wc_rk'))
    KP_sc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_rk'))
    KP_uc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_rk'))
    src_problems = [KP_uc_rk, KP_sc_rk, KP_wc_rk, KP_sc_ak] # configuration A cope with big data paper
    target_problem = KP_uc_ak
  elif src_version == 'kp1000-250-a':
    src_problem_set = [(250, 'KP_uc_rk'), (250, 'KP_sc_rk'), (250, 'KP_wc_rk'), (250, 'KP_sc_ak')] # Counter-Problem list
    for problem_num, problem_name in src_problem_set:
      for i in range(problem_num):
        src_problems.append(Tools.load_from_file(os.path.join(knapsack_problem_path, '{}{}'.format(problem_name, i))))
    KP_uc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_ak'))
    KP_sc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_ak'))
    target_problem = KP_uc_ak   
  elif src_version == 'kp1000-40-a':
    src_problem_set = [(320, 'KP_uc_rk'), (320, 'KP_sc_rk'), (320, 'KP_wc_rk'), (40, 'KP_sc_ak')] # Counter-Problem list
    for problem_num, problem_name in src_problem_set:
      for i in range(problem_num):
        src_problems.append(Tools.load_from_file(os.path.join(knapsack_problem_path, '{}{}'.format(problem_name, i))))
    KP_uc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_ak'))
    KP_sc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_ak'))
    target_problem = KP_uc_ak
  
  elif src_version == 'kp1000-10-a':
    src_problem_set = [(330, 'KP_uc_rk'), (330, 'KP_sc_rk'), (330, 'KP_wc_rk'), (10, 'KP_sc_ak')] # Counter-Problem list
    for problem_num, problem_name in src_problem_set:
      for i in range(problem_num):
        src_problems.append(Tools.load_from_file(os.path.join(knapsack_problem_path, '{}{}'.format(problem_name, i))))
    KP_uc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_ak'))
    KP_sc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_ak'))
    target_problem = KP_uc_ak
  elif src_version == 'kp1000-1-a':
    src_problem_set = [(333, 'KP_uc_rk'), (333, 'KP_sc_rk'), (333, 'KP_wc_rk'), (1, 'KP_sc_ak')] # Counter-Problem list
    for problem_num, problem_name in src_problem_set:
      for i in range(problem_num):
        src_problems.append(Tools.load_from_file(os.path.join(knapsack_problem_path, '{}{}'.format(problem_name, i))))
    KP_uc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_ak'))
    KP_sc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_ak'))
    target_problem = KP_uc_ak
  elif src_version == 'kp4-b':
    KP_uc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_ak'))
    KP_wc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_wc_ak'))
    KP_wc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_wc_rk'))
    KP_sc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_rk'))
    KP_uc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_rk'))
    src_problems = [KP_uc_rk, KP_sc_rk, KP_wc_rk, KP_uc_ak] # configuration B cope with big data paper
    target_problem = KP_wc_ak
  elif src_version == 'kp1000-250-b':
    src_problem_set = [(250, 'KP_uc_rk'), (250, 'KP_sc_rk'), (250, 'KP_wc_rk'), (250, 'KP_uc_ak')] # Counter-Problem list
    for problem_num, problem_name in src_problem_set:
      for i in range(problem_num):
        src_problems.append(Tools.load_from_file(os.path.join(knapsack_problem_path, '{}{}'.format(problem_name, i))))
    KP_wc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_wc_ak'))
    target_problem = KP_wc_ak   
  elif src_version == 'kp1000-40-b':
    src_problem_set = [(320, 'KP_uc_rk'), (320, 'KP_sc_rk'), (320, 'KP_wc_rk'), (40, 'KP_uc_ak')] # Counter-Problem list
    for problem_num, problem_name in src_problem_set:
      for i in range(problem_num):
        src_problems.append(Tools.load_from_file(os.path.join(knapsack_problem_path, '{}{}'.format(problem_name, i))))
    KP_wc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_wc_ak'))
    target_problem = KP_wc_ak
  elif src_version == 'kp4-c':
    KP_sc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_ak'))
    KP_wc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_wc_ak'))
    KP_wc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_wc_rk'))
    KP_sc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_rk'))
    KP_uc_rk = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_uc_rk'))
    src_problems = [KP_uc_rk, KP_sc_rk, KP_wc_rk, KP_wc_ak] # configuration C cope with big data paper
    target_problem = KP_sc_ak
  elif src_version == 'kp1000-250-c':
    src_problem_set = [(250, 'KP_uc_rk'), (250, 'KP_sc_rk'), (250, 'KP_wc_rk'), (250, 'KP_wc_ak')] # Counter-Problem list
    for problem_num, problem_name in src_problem_set:
      for i in range(problem_num):
        src_problems.append(Tools.load_from_file(os.path.join(knapsack_problem_path, '{}{}'.format(problem_name, i))))
    KP_sc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_ak'))
    target_problem = KP_sc_ak   
  elif src_version == 'kp1000-40-c':
    src_problem_set = [(320, 'KP_uc_rk'), (320, 'KP_sc_rk'), (320, 'KP_wc_rk'), (40, 'KP_wc_ak')] # Counter-Problem list
    for problem_num, problem_name in src_problem_set:
      for i in range(problem_num):
        src_problems.append(Tools.load_from_file(os.path.join(knapsack_problem_path, '{}{}'.format(problem_name, i))))
    KP_sc_ak = Tools.load_from_file(os.path.join(knapsack_problem_path, 'KP_sc_ak'))
    target_problem = KP_sc_ak 
  else:
    print('Source problems version is not correct {}'.format(src_version))

  print("All source problems & target problems are loaded: length = {}".format(len(src_problems)))

  src_models = []

  if buildmodel:
    # build source probabilistic models
    now = time()
    for problem in src_problems:
      src_models, _, _, _ = evolutionary_algorithm(problem, 1000, src_models=src_models, stop_condition=stop_condition)

    Tools.save_to_file(source_models_path + '_{}'.format(src_version), src_models)
    print('Building models took {} minutes'.format(str((time()-now)/60)))
    print('{} number of source models have been created.'.format(len(src_models)))
  else:
    try:
      src_models = Tools.load_from_file(source_models_path + '_{}'.format(src_version))
    except FileNotFoundError:
      print('Source models not exist in the {} path'.format(source_models_path))  

  return src_models, target_problem

def fill_max(arr):
    out_arr = np.zeros_like(arr)
    max_val = float('-inf')
    for i, e in enumerate(arr):
        if e > max_val:
            max_val = e
        out_arr[i] = max_val
    return out_arr