# -*- coding: utf-8 -*- 
# %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of Genetic Algorithm-based WaveLength Selection
# using Support Vector Regression (GAWLSSVR)

import random

import numpy as np
import pandas as pd
from deap import base
from deap import creator
from deap import tools
from sklearn import model_selection
from sklearn import svm

# settings
number_of_areas = 5
max_width_of_areas = 20

number_of_population = 100
number_of_generation = 150

svr_c_2_range = (-5, 10)
svr_epsilon_2_range = (-10, 0)
svr_gamma_2_range = (-20, 10)

fold_number = 5
probability_of_crossover = 0.5
probability_of_mutation = 0.2
threshold_of_variable_selection = 0.5

# load dataset
dataset = pd.read_csv('shootout_2002_train1.csv')
dataset = np.array(dataset)
x_train = dataset[:, 1:]
y_train = dataset[:, 0]

# autoscaling
autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)

# GAWLSSVR
creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # for minimization, set weights as (-1.0,)
creator.create('Individual', list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
min_boundary = np.zeros(number_of_areas * 2 + 3)
max_boundary = np.ones(number_of_areas * 2 + 3) * x_train.shape[1]
max_boundary[np.arange(1, number_of_areas * 2, 2)] = max_width_of_areas
min_boundary[-3] = svr_c_2_range[0]
min_boundary[-2] = svr_epsilon_2_range[0]
min_boundary[-1] = svr_gamma_2_range[0]
max_boundary[-3] = svr_c_2_range[1]
max_boundary[-2] = svr_epsilon_2_range[1]
max_boundary[-1] = svr_gamma_2_range[1]


def create_ind_uniform(min_boundary, max_boundary):
    index = []
    for min, max in zip(min_boundary, max_boundary):
        index.append(random.uniform(min, max))
    return index


toolbox.register('create_ind', create_ind_uniform, min_boundary, max_boundary)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.create_ind)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)


def evalOneMax(individual):
    individual_array = np.array(individual)
    individual_array_wavelength = np.array(np.floor(individual_array[0:number_of_areas * 2]), dtype=int)
    selected_x_variable_numbers = np.zeros(0, dtype=int)
    for area_number in range(number_of_areas):
        if individual_array_wavelength[2 * area_number] + individual_array_wavelength[2 * area_number + 1] <= \
                autoscaled_x_train.shape[1]:
            selected_x_variable_numbers = np.r_[
                selected_x_variable_numbers, np.arange(individual_array_wavelength[2 * area_number],
                                                       individual_array_wavelength[2 * area_number] +
                                                       individual_array_wavelength[2 * area_number + 1])]
        else:
            selected_x_variable_numbers = np.r_[
                selected_x_variable_numbers, np.arange(individual_array_wavelength[2 * area_number],
                                                       autoscaled_x_train.shape[1])]

        selected_autoscaled_x_train = autoscaled_x_train[:, selected_x_variable_numbers]
    if len(selected_x_variable_numbers):
        # cross-validation
        model_in_cv = svm.SVR(kernel='rbf', C=2 ** round(individual_array[-3]),
                              epsilon=2 ** round(individual_array[-2]), gamma=2 ** round(individual_array[-1]))
        estimated_y_train_in_cv = model_selection.cross_val_predict(model_in_cv, selected_autoscaled_x_train,
                                                                    autoscaled_y_train, cv=fold_number)
        estimated_y_train_in_cv = estimated_y_train_in_cv * y_train.std(ddof=1) + y_train.mean()
        value = 1 - sum((y_train - estimated_y_train_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)
    else:
        value = -999

    return value,


toolbox.register('evaluate', evalOneMax)
toolbox.register('mate', tools.cxTwoPoint)
toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
toolbox.register('select', tools.selTournament, tournsize=3)

# random.seed(100)
random.seed()
pop = toolbox.population(n=number_of_population)

print('Start of evolution')

fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

print('  Evaluated %i individuals' % len(pop))

for generation in range(number_of_generation):
    print('-- Generation {0} --'.format(generation + 1))

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < probability_of_crossover:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < probability_of_mutation:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print('  Evaluated %i individuals' % len(invalid_ind))

    pop[:] = offspring
    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print('  Min %s' % min(fits))
    print('  Max %s' % max(fits))
    print('  Avg %s' % mean)
    print('  Std %s' % std)

print('-- End of (successful) evolution --')

best_individual = tools.selBest(pop, 1)[0]
best_individual_array = np.array(best_individual)
best_individual_array_wavelength = np.array(np.floor(best_individual_array[0:number_of_areas * 2]), dtype=int)
selected_x_variable_numbers = np.zeros(0, dtype=int)
for area_number in range(number_of_areas):
    if best_individual_array_wavelength[2 * area_number] + best_individual_array_wavelength[2 * area_number + 1] <= \
            autoscaled_x_train.shape[1]:
        selected_x_variable_numbers = np.r_[
            selected_x_variable_numbers, np.arange(best_individual_array_wavelength[2 * area_number],
                                                   best_individual_array_wavelength[2 * area_number] +
                                                   best_individual_array_wavelength[2 * area_number + 1])]
    else:
        selected_x_variable_numbers = np.r_[
            selected_x_variable_numbers, np.arange(best_individual_array_wavelength[2 * area_number],
                                                   autoscaled_x_train.shape[1])]

print('Selected variables : %s, %s' % (selected_x_variable_numbers, best_individual.fitness.values))
print('C : 2 ** {0}'.format(round(best_individual_array[-3])))
print('Epsilon : 2 ** {0}'.format(round(best_individual_array[-2])))
print('Gamma : 2 ** {0}'.format(round(best_individual_array[-1])))
