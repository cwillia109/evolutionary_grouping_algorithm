from numba import njit
import pandas as pd
import numpy as np
from random import uniform
from tqdm import tqdm
from math import sqrt


df = pd.read_csv(r"C:\Users\Corey\OneDrive - University of Toledo\SRN DSGN\py-data.csv")
preference_columns = []
for x in range(8):
    preference_columns.append(str(x))


preference_matrix = df[preference_columns].to_numpy()
required = df["Required"].to_numpy()
#required[:] = np.nan
restricted_columns = ['R1', 'R2', 'R3', 'R4']
restricted = df[restricted_columns].to_numpy()
restricted = np.empty(np.shape(restricted))
restricted[:][:] = np.nan


projects = np.array(pd.unique(df[preference_columns].values.ravel()), dtype='float')
projects = np.sort(projects)
class_size = np.size(preference_matrix, 0)



@njit
def init_population(max_population, num_students, projects, req):
    population = np.zeros((max_population, num_students))
    for i in range(max_population):
        for j in range(num_students):
            if np.isnan(req[j]): 
                population[i][j] = np.random.choice(projects)
            else:
                population[i][j] = req[j]
    return population


@njit
def cost(population, pm):
    max_population = np.shape(population)[0]
    num_students = np.shape(pm)[0]
    num_of_choices = np.shape(pm)[1]
    t_cost = np.zeros(max_population)
    for i in range(max_population):
        solution = population[i].copy()
        cost = 0
        for j in range(num_students):
            choice = np.where(preference_matrix[j] == solution[j])[0]
            # Some students had single projects selected multiple times or not at all
            if len(choice) == 0:
                choice = num_of_choices
            elif len(choice) > 1:
                choice = np.min(choice)
            else:
                choice = choice.item()
            cost += choice
        t_cost[i] = cost
    return t_cost


@njit
def iterate_pop(popn, pm, projects, req, it, itmax):
    population = popn.copy()
    solutions = np.shape(population)[0]
    num_of_students = np.shape(population)[1]
    num_of_choices = np.shape(pm)[1]
    # Loop through each solution
    for i in range(solutions):
        offspring = population[i].copy()
        # Loop through each allele
        parent_cost = 0.0
        offspring_cost = 0.0
        for j in range(num_of_students):
            satisfaction = np.where(pm[j] == offspring[j])[0]
            # Some students had single projects selected multiple times or not at all
            if len(satisfaction) == 0:
                satisfaction = num_of_choices
            elif len(satisfaction) > 1:
                satisfaction = np.amin(satisfaction)
            else:
                satisfaction = satisfaction.item()

            parent_cost += satisfaction

            instances = np.sum(offspring == offspring[j])
            # Discourage over grouping on a single project
            group_count = ((instances%4)%3) if instances <= 12 else 5
            # Don't allow the required student to change
            if offspring[j] == req[j]:
                p = 0
            else:
                p = (satisfaction**2 + group_count**2) / (num_of_choices**2 - group_count**2)
            
            if uniform(0,1) < p:
                new_allele = np.random.choice(projects)
                offspring[j] = new_allele
                satisfaction = np.where(pm[j] == new_allele)[0]
                # Get new allele satisfaction
                if len(satisfaction) == 0:
                    satisfaction = num_of_choices
                elif len(satisfaction) > 1:
                    satisfaction = np.min(satisfaction)
                else:
                    satisfaction = satisfaction.item()
                offspring_cost += satisfaction
            else:
                offspring_cost += satisfaction
        
        if offspring_cost < parent_cost:
            population[i] = offspring.copy()
    return population


new_pop = init_population(100, class_size, projects, required)
starting_cost = cost(new_pop, preference_matrix)
itermax = 10000
it = 0
F_min_arr = np.zeros(itermax)
F_avg_arr = np.zeros(itermax)
for _ in tqdm(range(itermax), ncols=100):
    new_pop = iterate_pop(new_pop, preference_matrix, projects, required, it, itermax)
    f_cost = cost(new_pop, preference_matrix)
    F_min_arr[_] = f_cost.min()
    F_avg_arr[_] = f_cost.mean()
    it += 1

ending_cost = cost(new_pop, preference_matrix)
print(" - - - - Evolutionary Strategies - - - - ")
print("Starting cost: ", int(starting_cost.min()))
print("Ending cost: ", int(ending_cost.min()))
spare_seats = 0
best_sol = new_pop[f_cost.argmin()]
instance_arr = []
for proj in projects:
    instances = np.sum(best_sol == proj)
    instance_arr.append(instances)
    alpha = (instances%4)%3
    spare_seats += alpha
print("Spare_seats: ", spare_seats)
print("Largest team: ", max(instance_arr), "Smallest Team: ", min(instance_arr))
data = {'min': F_min_arr,
        'avg': F_avg_arr
    }
df = pd.DataFrame(data, columns=["min", "avg"])
df.to_csv("eva.csv")