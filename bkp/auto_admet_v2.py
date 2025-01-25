import warnings
warnings.filterwarnings("ignore")
import time
import multiprocessing
import alogos as al
import random
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFpr, SelectFwe, SelectFdr, chi2, f_classif
import pandas as pd
import numpy as np
import argparse
from pgmpy.estimators import HillClimbSearch, K2Score
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

def generate_sentence(grammar, start_symbol, seed, max_depth=10):
    if max_depth == 0:
        return ""

    if start_symbol not in grammar:
        return start_symbol

    expansion = random.choice(grammar[start_symbol])
    sentence = ""
    for symbol in expansion:
        if symbol in grammar:
            sentence += generate_sentence(grammar, symbol, seed, max_depth - 1)
        else:
            sentence += symbol
    return sentence

def generate_population(grammar, start_symbol, population_size, seed=None):
    population = []
    for _ in range(population_size):
        population.append(generate_sentence(grammar, start_symbol, seed, 10))
    return population

def evolve(population, grammar, start_symbol, mutation_rate, crossover_rate, dataset_path, time_budget_minutes_alg_eval, num_cores, resample, generation, seed, max_retries=100):
    pop_fitness_scores = evaluate_population_parallel(population, dataset_path, time_budget_minutes_alg_eval, num_cores, resample, generation)
    population = pop_fitness_scores[0]
    fitness_scores = pop_fitness_scores[1]
    total_fitness = sum(fitness_scores)
    new_population = []

    print("Population Fitness:")
    for i in range(len(population)):
        print(f"{population[i]}, Fitness: {fitness_scores[i]:.4f}")

    # Select elites
    num_elites = 1
    elites_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)[:num_elites]
    elites = [population[i] for i in elites_indices]
    new_population.extend(elites)

    while len(new_population) < len(population):
        if random.random() < crossover_rate:
            parent1 = random.choices(population, weights=fitness_scores, k=1)[0]
            parent2 = random.choices(population, weights=fitness_scores, k=1)[0]
            child1, child2 = crossover(parent1, parent2, grammar, seed)
        else:
            child1, child2 = random.choices(population, k=2)

        retries = 0
        while retries < max_retries:
            if ("$$" not in child1 and "$#" not in child1 and child1):
                new_population.append(child1)
                break
            child1 = mutate(child1, grammar, start_symbol, seed, mutation_rate)
            retries += 1

        if len(new_population) < len(population):
            retries = 0
            while retries < max_retries:
                if ("$$" not in child2 and "$#" not in child2 and child2):
                    new_population.append(child2)
                    break
                child2 = mutate(child2, grammar, start_symbol, seed, mutation_rate)
                retries += 1

    return new_population

def update_grammar_with_bayesian_network(population, fitness_scores, grammar, top_n=5, bottom_n=5):
    """
    Update the grammar based on a Bayesian network learned from the top and bottom individuals.
    Include high-level hyperparameters in the learning process.
    """
    # Prepare data for Bayesian network
    sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)
    top_individuals = [population[i] for i in sorted_indices[:top_n]]
    bottom_individuals = [population[i] for i in sorted_indices[-bottom_n:]]

    # Combine top and bottom individuals for learning
    combined_data = top_individuals + bottom_individuals

    # Parse individuals into features for Bayesian network
    data = []
    for individual in combined_data:
        features = individual.split("#")
        representation, scaling, selection, algorithm = features[:4]
        high_level_params = features[4:] if len(features) > 4 else []
        row = {
            "Representation": representation,
            "Scaling": scaling,
            "Selection": selection,
            "Algorithm": algorithm,
        }
        # Add high-level hyperparameters as separate columns
        for i, param in enumerate(high_level_params):
            row[f"Param_{i}"] = param
        data.append(row)

    df = pd.DataFrame(data)

    # Learn Bayesian network
    hc = HillClimbSearch(df)
    best_model = hc.estimate(scoring_method=K2Score(df))

    # Use Bayesian network to update grammar
    for edge in best_model.edges():
        print(f"Dependency found: {edge[0]} -> {edge[1]}")
        if edge[0] in grammar and edge[1] in grammar:
            grammar[edge[0]].append(edge[1])

    return grammar

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoML with Bayesian Grammar Update")
    parser.add_argument("training_dir", help="Path to training dataset")
    parser.add_argument("output_dir", help="Output path for results")
    parser.add_argument("-pop_size", type=int, default=30, help="Population size")
    parser.add_argument("-xover_rate", type=float, default=0.9, help="Crossover rate")
    parser.add_argument("-mut_rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("-time_budget_min", type=int, default=60, help="Time budget in minutes")
    parser.add_argument("-seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    grammar = {
        "<Start>": [["<feature_definition>", "#", "<feature_scaling>", "#", "<feature_selection>", "#", "<algorithms>"]],
        "<feature_definition>": [["General_Descriptors"], ["Advanced_Descriptors"],
                             ["Graph_based_Signatures"],
                             ["Toxicophores"],
                             ["Fragments"],
                             ["General_Descriptors", "$", "Advanced_Descriptors"],
                             ["General_Descriptors", "$","Graph_based_Signatures"],
                             ["General_Descriptors", "$","Toxicophores"],
                             ["General_Descriptors", "$","Fragments"],
                             ["Advanced_Descriptors", "$","Graph_based_Signatures"],
                             ["Advanced_Descriptors", "$","Toxicophores"],
                             ["Advanced_Descriptors", "$","Fragments"],
                             ["Graph_based_Signatures", "$","Toxicophores"],
                             ["Graph_based_Signatures", "$","Fragments"],
                             ["Toxicophores", "$","Fragments"],
                             ["General_Descriptors", "$","Advanced_Descriptors", "$","Graph_based_Signatures"],
                             ["General_Descriptors", "$","Advanced_Descriptors", "$","Toxicophores"],
                             ["General_Descriptors", "$","Advanced_Descriptors", "$","Fragments"],
                             ["General_Descriptors", "$","Graph_based_Signatures", "$","Toxicophores"],
                             ["General_Descriptors", "$","Graph_based_Signatures", "$","Fragments"],
                             ["General_Descriptors", "$","Toxicophores", "$","Fragments"],
                             ["Advanced_Descriptors", "$","Graph_based_Signatures", "$","Toxicophores"],
                             ["Advanced_Descriptors", "$","Graph_based_Signatures", "$","Fragments"],
                             ["Advanced_Descriptors", "$","Toxicophores", "$","Fragments"],
                             ["Graph_based_Signatures", "$","Toxicophores", "$","Fragments"],
                             ["General_Descriptors", "$","Advanced_Descriptors", "$","Graph_based_Signatures", "$","Toxicophores"],
                             ["General_Descriptors", "$","Advanced_Descriptors", "$","Graph_based_Signatures", "$","Fragments"],
                             ["General_Descriptors", "$","Advanced_Descriptors", "$","Toxicophores", "$","Fragments"],
                             ["General_Descriptors", "$","Graph_based_Signatures", "$","Toxicophores", "$","Fragments"],
                             ["Advanced_Descriptors", "$","Graph_based_Signatures", "$","Toxicophores", "$","Fragments"],
                             ["General_Descriptors", "$","Advanced_Descriptors", "$","Graph_based_Signatures", "$","Toxicophores", "$","Fragments"]],
        "<feature_scaling>": [["<None>"], ["Normalizer", "$", "<norm>"], ["MinMaxScaler"], ["MaxAbsScaler"], ["RobustScaler", "$", "<boolean>", "$", "<boolean>"], ["StandardScaler", "$", "<boolean>", "$", "<boolean>"]],
        "<feature_selection>": [["<None>"], ["VarianceThreshold", "$", "<threshold>"], ["SelectPercentile", "$",  "<percentile>",  "$",  "<score_function>"],
                           ["SelectFpr", "$", "<value_rand_1>", "$", "<score_function>"], ["SelectFwe", "$", "<value_rand_1>", "$", "<score_function>"],
                            ["SelectFdr", "$", "<value_rand_1>", "$", "<score_function>"]],
        "<algorithms>": [["AdaBoostClassifier", "$", "<algorithm_ada>", "$", "<n_estimators>", "$", "<learning_rate_ada>"],
                     ["DecisionTreeClassifier", "$", "<criterion>", "$", "<splitter>", "$", "<max_depth>", "$", "<min_samples_split>", "$", "<min_samples_leaf>", "$", "<max_features>", "$", "<class_weight>"],
                     ["ExtraTreeClassifier", "$", "<criterion>", "$", "<splitter>", "$", "<max_depth>", "$", "<min_samples_split>", "$", "<min_samples_leaf>", "$", "$", "<max_features>", "$", "$", "<class_weight>"],
                     ["RandomForestClassifier","$", "<n_estimators>", "$", "<criterion>", "$", "<max_depth>", "$", "<min_samples_split>", "$", "<min_samples_leaf>", "$", "<max_features>", "$", "<class_weight_rf>"],
                     ["ExtraTreesClassifier","$", "<n_estimators>", "$", "<criterion>", "$", "<max_depth>", "$", "<min_samples_split>", "$", "<min_samples_leaf>", "$", "<max_features>", "$", "<class_weight_rf>"],
                     ["GradientBoostingClassifier","$", "<n_estimators>", "$", "<criterion_gb>", "$", "<max_depth>", "$", "<min_samples_split>", "$", "<min_samples_leaf>", "$", "<max_features>", "$", "<loss>"],
                     ["XGBClassifier", "$", "<n_estimators>", "$", "<max_depth>", "$", "<max_leaves>", "$", "<learning_rate_ada>"]
                    ],
        "<None>": [["None"]],
        "<norm>": [["l1"], ["l2"], ["max"]],
        "<threshold>": [["0.0"],["0.05"],["0.10"],["0.15"],["0.20"],["0.25"],["0.30"],["0.35"],["0.40"],["0.45"],["0.50"],
                    ["0.55"],["0.60"],["0.65"],["0.70"],["0.75"],["0.80"],["0.85"],["0.90"],["0.95"],["1.0"]],
        "<algorithm_ada>": [["SAMME.R"], ["SAMME"]],   
        "<n_estimators>": [["5"],["10"],["15"],["20"],["25"],["30"],["35"],["40"],["45"],["50"],["100"], ["150"], ["200"], ["250"], ["300"]],
        "<learning_rate_ada>": [["0.01"],["0.05"],["0.10"],["0.15"],["0.20"],["0.25"],["0.30"],["0.35"],["0.40"],["0.45"],
                            ["0.50"],["0.55"],["0.60"],["0.65"],["0.70"],["0.75"],["0.80"],["0.85"],["0.90"],["0.95"],["1.0"],
                            ["1.05"],["1.10"],["1.15"],["1.20"],["1.25"],["1.30"],["1.35"],["1.40"],["1.45"],["1.50"],
                            ["1.55"],["1.60"],["1.65"],["1.70"],["1.75"],["1.80"],["1.85"],["1.90"],["1.95"],["2.0"]],
        "<boolean>": [["True"], ["False"]],
        "<percentile>": [["5"],["10"],["15"],["20"],["25"],["30"],["35"],["40"],["45"],["50"],["55"],["60"],["65"],["70"],["75"],["80"],["85"],["90"],["95"]],
        "<score_function>": [["f_classif"], ["chi2"]],
        "<value_rand_1>": [["0.0"],["0.05"],["0.10"],["0.15"],["0.20"],["0.25"],["0.30"],["0.35"],["0.40"],["0.45"],["0.50"],["0.55"],["0.60"],["0.65"],["0.70"],["0.75"],["0.80"],["0.85"],["0.90"],["0.95"],["1.0"]],
        "<criterion>": [["gini"], ["entropy"], ["log_loss"]],
        "<splitter>": [["best"], ["random"]],
        "<max_depth>": [["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9"], ["10"], ["None"]],
        "<min_samples_split>": [["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9"], ["10"], ["11"], ["12"], ["13"], ["14"], ["15"], ["16"], ["17"], ["18"], ["19"], ["20"]],
        "<min_samples_leaf>": [["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9"], ["10"], ["11"], ["12"], ["13"], ["14"], ["15"], ["16"], ["17"], ["18"], ["19"], ["20"]],
        "<max_features>": [["None"], ["log2"], ["sqrt"]],
        "<class_weight>": [["balanced"], ["None"]],
        "<class_weight_rf>": [["balanced"], ["balanced_subsample"], ["None"]],
        "<criterion_gb>": [["friedman_mse"], ["squared_error"]],
       "<loss>": [["log_loss"], ["exponential"]],
        "<max_leaves>": [["0"], ["1"], ["2"], ["3"], ["4"], ["5"], ["6"], ["7"], ["8"], ["9"], ["10"]]
    }

    start_symbol = "<Start>"
    population_size = args.pop_size
    seed = args.seed
    random.seed(seed)
    seed_sampling = 1
	
    population = generate_population(grammar, start_symbol, population_size, seed)
    training_set_path = args.training_dir
    testing_set_path = args.testing_dir
    start_time = time.time()
    elapsed_time = 0

    # Convert time budget from minutes to seconds
    time_budget_min = args.time_budget_min
    time_budget_seconds = time_budget_min * 60
    time_budget_minutes_alg_eval = args.time_budget_minutes_alg_eval
    num_cores = args.num_cores

    generation = 0
    resample = False
    while elapsed_time < time_budget_seconds:
        print("Generation", generation)

        population = evolve(population, grammar, start_symbol, args.mut_rate, args.xover_rate, training_set_path, time_budget_minutes_alg_eval, num_cores, resample, seed_sampling, seed)
        elapsed_time = time.time() - start_time
        generation += 1
        if generation % 5 == 0:
        	resample = True
        	seed_sampling += seed_sampling
        else:
        	resample = False	    

    best_pipeline = max(population, key=lambda pipeline: fitness_function(pipeline, training_set_path, False, generation))
    best_fitness_5CV = fitness_function(best_pipeline, training_set_path, False, generation)
    best_fitness_test = fitness_function_train_test(best_pipeline, training_set_path, testing_set_path)
    end_time = time.time()    

    file = open(args.output_dir, "a")
    file.write("seed,elapsed_time,generation,best_pipeline,result_cv,result_test\n")
    file.write(str(seed) + "," + str(elapsed_time) + "," + str(generation) + "," + str(best_pipeline) + "," + str(best_fitness_5CV) + "," + str(best_fitness_test) + "\n")
    file.close()       

