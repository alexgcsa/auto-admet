import argparse
import random

from src.grammar_boa_gp import GrammarBayesOptGeneticfProgAlgorithm
from src.bnf_grammar_parser import BNFGrammar



# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")

    # Mandatory arguments
    parser.add_argument("training_path", type=str, help="Path to the training dataset.")
    parser.add_argument("testing_path", type=str, help="Path to the training dataset.")
    parser.add_argument("grammar_path", type=str, help="Path to the grammar defining the AutoML search space.")
    parser.add_argument("output_dir", type=str, help="Output directory.")
    
    # Optional arguments 
    parser.add_argument("-s", "--seed", type=int, help="The seed", default=1) 
    parser.add_argument("-m", "--metric", type=str, help="The metric to be used during biochemical property predicion optimisation procedure", default="auc")
    parser.add_argument("-e", "--exp_name", type=str, help="The name of the experiment", default="Exp_ADMET")
    parser.add_argument("-t", "--time", type=int, help="Time in minutes to run the method", default=5)
    parser.add_argument("-n", "--ncores", type=int, help="Number of cores", default=20)
    parser.add_argument("-ta", "--time_budget_minutes_alg_eval", type=int, help="Time to assess each individual (i.e., ML pipeline)", default=1)
    parser.add_argument("-p", "--population_size", type=int, help="Population size", default=100)
    parser.add_argument("-mr", "--mutation_rate", type=float, help="Mutation rate", default=0.15)
    parser.add_argument("-cr", "--crossover_rate", type=float, help="Crossover rate", default=0.80)
    parser.add_argument("-cmr", "--crossover_mutation_rate", type=float, help="Crossover and mutation rate", default=0.05)
    parser.add_argument("-es", "--elitism_size", type=int, help="Elitism size", default=1)

    # Parse arguments
    args = parser.parse_args()
    training_path = args.training_path
    testing_path = args.training_path
    grammar_path = args.grammar_path
    
    seed = args.seed
    metric = args.metric
    expname = args.exp_name
    maxtime = args.time
    ncores = args.ncores
    timebudgetminutesalgeval = args.time_budget_minutes_alg_eval
    populationsize = args.population_size
    mutationrate = args.mutation_rate
    crossoverrate = args.crossover_rate
    crossovermutationrate = args.crossover_mutation_rate
    elitismsize = args.elitism_size

    
    random.seed(seed)  # For reproducibility

    # Define grammar
    with open(grammar_path, "r") as file:
        grammar_text = file.read()

    # Load grammar
    grammar = BNFGrammar()
    grammar.load_grammar(grammar_text)

    # Run GGP
    ggp = GrammarBayesOptGeneticfProgAlgorithm(grammar, training_path, testing_path, fitness_metric=metric, experiment_name=expname, seed=seed, max_time=maxtime, 
                                                num_cores=ncores, time_budget_minutes_alg_eval=timebudgetminutesalgeval, population_size = populationsize,
                                                mutation_rate = mutationrate, crossover_rate = crossoverrate, crossover_mutation_rate = crossovermutationrate,
                                                elitism_size = elitismsize)
    best_program = ggp.evolve()
