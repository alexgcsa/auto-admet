import random
from collections import defaultdict
from copy import deepcopy


class BNFGrammar:
    def __init__(self):
        self.grammar = defaultdict(list)
        self.non_terminals = set()
        self.terminals = set()

    def load_grammar(self, bnf_text: str):
        """
        Parses the BNF grammar from a string.
        """
        for line in bnf_text.strip().splitlines():
            if "::=" in line:
                lhs, rhs = line.split("::=", 1)
                lhs = lhs.strip()
                self.non_terminals.add(lhs)
                rhs_options = [option.strip() for option in rhs.split("|")]
                for option in rhs_options:
                    self.grammar[lhs].append(option.split())
                    for token in option.split():
                        if token not in self.non_terminals:
                            self.terminals.add(token)

    def generate_parse_tree(self, symbol: str = "<start>", max_depth: int = 10) -> dict:
        """
        Generates a parse tree starting from the given symbol, ensuring mandatory grammar components are included.
        """
        if max_depth <= 0 or symbol not in self.grammar:
            return symbol  # Return the symbol as a terminal if max depth is reached
    
        # Strictly enforce the `<start>` rule
        if symbol == "<start>":
            # Generate each mandatory component
            feature_def = self.generate_parse_tree("<feature_definition>", max_depth - 1)
            scaling = self.generate_parse_tree("<feature_scaling>", max_depth - 1)
            selection = self.generate_parse_tree("<feature_selection>", max_depth - 1)
            ml_algo = self.generate_parse_tree("<ml_algorithms>", max_depth - 1)
    
            return {symbol: [feature_def, "#", scaling, "#", selection, "#", ml_algo]}
    
        # Select a random production for other non-terminals
        production = random.choice(self.grammar[symbol])
        return {symbol: [self.generate_parse_tree(token, max_depth - 1) for token in production]}
   

    def parse_tree_to_string(self, tree) -> str:
        """
        Reconstructs a string from the parse tree.
        """
        if isinstance(tree, str):
            # Leaf node (terminal)
            return tree
        # Non-terminal with its production rules as children
        root, children = list(tree.items())[0]
        return " ".join(self.parse_tree_to_string(child) for child in children)

    def validate_parse_tree(self, tree, symbol="<start>") -> bool:
        """
        Validates if the parse tree conforms to the grammar and respects the `<start>` structure.
        """
        if isinstance(tree, str):
            return tree in self.terminals  # Check terminal validity
    
        if not isinstance(tree, dict) or len(tree) != 1:
            return False
    
        root, children = list(tree.items())[0]
        if root != symbol:
            return False
    
        if symbol == "<start>":
            # Check `<start>` structure
            if len(children) != 7:
                return False
            expected_symbols = ["<feature_definition>", "#", "<feature_scaling>", "#", "<feature_selection>", "#", "<ml_algorithms>"]
            for i, child_symbol in enumerate(expected_symbols):
                if i % 2 == 0 and not self.validate_parse_tree(children[i], child_symbol):  # Validate non-terminals
                    return False
                if i % 2 == 1 and children[i] != "#":  # Ensure separator
                    return False
    
        # Validate other non-terminals
        for production in self.grammar[symbol]:
            if len(production) == len(children) and all(
                self.validate_parse_tree(child, production[i])
                for i, child in enumerate(children)
            ):
                return True
    
        return False


class GrammarBasedGP:
    def __init__(self, grammar, population_size=100, max_generations=20, mutation_rate=0.1, crossover_rate=0.7, crossover_mutation_rate=0.05, elitism_size=1):
        self.grammar = grammar
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_mutation_rate = crossover_mutation_rate
        self.elitism_size = elitism_size
        self.population = []

    def fitness(self, individual):
        """
        Fitness function: minimize the size of the parse tree.
        """
        return self.count_nodes(individual)

    def count_nodes(self, tree):
        """
        Counts the number of nodes in the parse tree.
        """
        if isinstance(tree, str):  # Terminal node
            return 1
        return 1 + sum(self.count_nodes(child) for child in list(tree.values())[0])
    
    def crossover(self, parent1, parent2):
        """
        Performs crossover by swapping compatible components between parents.
        """
        if isinstance(parent1, str) or isinstance(parent2, str):  # No crossover if terminal
            return parent1, parent2
    
        root1, children1 = list(parent1.items())[0]
        root2, children2 = list(parent2.items())[0]
    
        if root1 == "<start>" and root2 == "<start>":
            # Swap one of the four main components of `<start>`
            idx = random.choice([0, 2, 4, 6])  # Indices of the main components
            children1[idx], children2[idx] = children2[idx], children1[idx]
        elif root1 == root2:
            # Swap subtrees for other non-terminals
            idx1 = random.randint(0, len(children1) - 1)
            idx2 = random.randint(0, len(children2) - 1)
            children1[idx1], children2[idx2] = children2[idx2], children1[idx1]
    
        return parent1, parent2




    
    def mutate(self, individual, max_mutation_depth=4):
        """
        Mutates an individual by replacing a specific component with a new valid subtree.
        """
        if isinstance(individual, str):  # Terminal, no mutation possible
            return individual
    
        root, children = list(individual.items())[0]
    
        if root == "<start>":
            # Mutate one of the four main components of `<start>`
            idx = random.choice([0, 2, 4, 6])  # Indices of the main components
            components = ["<feature_definition>", "<feature_scaling>", "<feature_selection>", "<ml_algorithms>"]
            replacement = self.generate_parse_tree(components[idx // 2], max_depth=max_mutation_depth)
            children[idx] = replacement
        else:
            # Mutate other non-terminals
            idx = random.randint(0, len(children) - 1)
            children[idx] = self.generate_parse_tree(root, max_depth=max_mutation_depth)
    
        return individual



    def generate_parse_tree(self, symbol: str = "<start>", max_depth: int = 10) -> dict:
        """
        Generates a parse tree starting from the given symbol, ensuring mandatory grammar components are included.
        """
        if max_depth <= 0 or symbol not in self.grammar:
            return symbol  # Return the symbol as a terminal if max depth is reached
    
        # Strictly enforce the `<start>` rule
        if symbol == "<start>":
            # Generate each mandatory component
            feature_def = self.generate_parse_tree("<feature_definition>", max_depth - 1)
            scaling = self.generate_parse_tree("<feature_scaling>", max_depth - 1)
            selection = self.generate_parse_tree("<feature_selection>", max_depth - 1)
            ml_algo = self.generate_parse_tree("<ml_algorithms>", max_depth - 1)
    
            return {symbol: [feature_def, "#", scaling, "#", selection, "#", ml_algo]}
    
        # Select a random production for other non-terminals
        production = random.choice(self.grammar[symbol])
        return {symbol: [self.generate_parse_tree(token, max_depth - 1) for token in production]}






    def evolve(self):
        """
        Runs the genetic programming algorithm.
        """
        # Initialize population
        self.population = [self.grammar.generate_parse_tree() for _ in range(self.population_size)]
        for s in self.population:
            print(grammar.parse_tree_to_string(s))
            
        print("-----------------------------------------------")
        for generation in range(self.max_generations):
            # Evaluate fitness
            fitness_scores = [(self.fitness(ind), ind) for ind in self.population]
            fitness_scores.sort(key=lambda x: x[0])

            # Elitism: retain the best individuals
            new_population = [ind for _, ind in fitness_scores[:self.elitism_size]]

            # Selection probabilities
            fitness_values = [1.0 / (f + 1e-6) for f, _ in fitness_scores]
            total_fitness = sum(fitness_values)
            probabilities = [f / total_fitness for f in fitness_values]

            while len(new_population) < self.population_size:
                random_num = random.random()                
                parent1, parent2 = random.choices(self.population, probabilities, k=2)
                
                if (random_num < self.crossover_mutation_rate):                    
                    #perform crossover
                    child1, child2 = self.crossover(deepcopy(parent1), deepcopy(parent2))                    
                    # and mutation                    
                    child1_1 = self.mutate(deepcopy(child1))
                    child2_1 = self.mutate(deepcopy(child2))
                    new_population.extend([child1_1, child2_1]) 
                elif (random_num < (self.crossover_mutation_rate + self.mutation_rate)):
                    #only mutation
                    child = self.mutate(deepcopy(parent1))
                    new_population.append(child)
                elif (random_num < (self.crossover_mutation_rate + self.mutation_rate + self.crossover_rate)):
                    #only crossover
                    child1, child2 = self.crossover(deepcopy(parent1), deepcopy(parent2))
                    new_population.extend([child1, child2])     
                else:
                    #no operation
                    new_population.extend([deepcopy(parent1), deepcopy(parent2)])
            
       

        

            # Trim excess individuals
            self.population = new_population[:self.population_size]

            for s in self.population:
                print()
                print(grammar.parse_tree_to_string(s))

            print("-----------------------------------------------")            

            # Print best individual of the generation
            best_fitness, best_individual = fitness_scores[0]
            #print(f"Generation {generation}: Best Fitness = {best_fitness}")
            #print(f"Best Individual: {self.grammar.parse_tree_to_string(best_individual)}")

        return fitness_scores[0][1]  # Return the best individual


# Example Usage
if __name__ == "__main__":
    random.seed(5)  # For reproducibility

    # Define grammar
    grammar_text = """
    <start> ::= <feature_definition> # <feature_scaling> # <feature_selection> # <ml_algorithms>
    <feature_definition> ::=  General_Descriptors | Advanced_Descriptors | Graph_based_Signatures | Toxicophores | Fragments | General_Descriptors Advanced_Descriptors | General_Descriptors Graph_based_Signatures | General_Descriptors Toxicophores | General_Descriptors Fragments | Advanced_Descriptors Graph_based_Signatures | Advanced_Descriptors Toxicophores | Advanced_Descriptors Fragments | Graph_based_Signatures Toxicophores | Graph_based_Signatures Fragments | Toxicophores Fragments | General_Descriptors Advanced_Descriptors Graph_based_Signatures | General_Descriptors Advanced_Descriptors Toxicophores | General_Descriptors Advanced_Descriptors Fragments | General_Descriptors Graph_based_Signatures Toxicophores | General_Descriptors Graph_based_Signatures Fragments | General_Descriptors Toxicophores Fragments | Advanced_Descriptors Graph_based_Signatures Toxicophores | Advanced_Descriptors Graph_based_Signatures Fragments | Advanced_Descriptors Toxicophores Fragments | Graph_based_Signatures Toxicophores Fragments | General_Descriptors Advanced_Descriptors Graph_based_Signatures Toxicophores | General_Descriptors Advanced_Descriptors Graph_based_Signatures Fragments | General_Descriptors Advanced_Descriptors Toxicophores Fragments | General_Descriptors Graph_based_Signatures Toxicophores Fragments | Advanced_Descriptors Graph_based_Signatures Toxicophores Fragments | General_Descriptors Advanced_Descriptors Graph_based_Signatures Toxicophores Fragments
    <feature_scaling> ::= <none_scaling> | <normalizer> | MinMaxScaler | MaxAbsScaler | <robust_scaler> | <standard_scaler>
    <normalizer> ::= Normalizer <norm>
    <robust_scaler> ::= RobustScaler <boolean> <boolean>
    <standard_scaler> ::= StandardScaler <boolean> <boolean>
    <feature_selection> ::= <none_feature_selection> | <variance_threshold> | <select_percentile> | <selectfpr> | <selectfwe> | <selectfdr>
    <variance_threshold> ::= VarianceThreshold <threshold>
    <select_percentile> ::= SelectPercentile <percentile> <score_function>
    <selectfpr> ::= SelectFpr <value_rand_1> <score_function>
    <selectfwe> ::= SelectFwe <value_rand_1> <score_function>
    <selectfdr> ::= SelectFdr <value_rand_1> <score_function>
    <ml_algorithms> ::= <adaboost> | <decision_tree> | <extra_tree> | <random_rorest> | <extra_trees> | <gradient_boosting> |  <xgboost>
    <adaboost> ::= AdaBoostClassifier <algorithm_ada> <n_estimators> <learning_rate_ada>
    <decision_tree> ::= DecisionTreeClassifier <criterion> <splitter> <max_depth> <min_samples_split> <min_samples_leaf> <max_features> <class_weight>
    <extra_tree> ::= ExtraTreeClassifier <criterion> <splitter> <max_depth> <min_samples_split> <min_samples_leaf> <max_features> <class_weight> 
    <random_rorest> ::= RandomForestClassifier <n_estimators> <criterion> <max_depth> <min_samples_split> <min_samples_leaf> <max_features> <class_weight_rf>
    <extra_trees> ::= ExtraTreesClassifier <n_estimators> <criterion> <max_depth> <min_samples_split> <min_samples_leaf> <max_features> <class_weight_rf>
    <gradient_boosting> ::= GradientBoostingClassifier <n_estimators> <criterion_gb> <max_depth> <min_samples_split> <min_samples_leaf> <max_features> <loss>
    <xgboost> ::= XGBClassifier <n_estimators> <max_depth> <max_leaves> <learning_rate_ada>
    <none_scaling> ::= None_scaling
    <none_feature_selection> ::= None_feature_selection
    <norm> ::= l1 | l2 | max
    <threshold> ::= 0.0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 1.0
    <algorithm_ada> ::= SAMME.R | SAMME
    <n_estimators> ::= 2 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 45 | 50 | 55 | 60 | 65 | 70 | 75 | 80 | 90 | 95 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 600 | 700 | 900 | 1000 | 1500 | 2000 | 2500 | 3000 | 4000 | 5000 | 7500 | 10000
    <learning_rate_ada> ::= 0.01 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 1.0 | 1.05 | 1.10 | 1.15 | 1.20 | 1.25 | 1.30 | 1.35 | 1.40 | 1.45 | 1.50 | 1.55 | 1.60 | 1.65 | 1.70 | 1.75 | 1.80 | 1.85 | 1.90 | 1.95 | 2.0
    <boolean> ::= True | False
    <percentile> ::= 5 | 10 | 15 | 20 | 25 | 30 | 35 | 45 | 50 | 55 | 60 | 65 | 70 | 75 | 80 | 90 | 95
    <score_function> ::= f_classif | chi2
    <value_rand_1> ::= 0.0 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.45 | 0.50 | 0.55 | 0.60 | 0.65 | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 | 0.95 | 1.0
    <criterion> ::= gini | entropy | log_loss
    <splitter> ::= best | random
    <max_depth> ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | None
    <min_samples_split> ::= 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20
    <min_samples_leaf> ::= 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20
    <max_features> ::= None | log2 | sqrt
    <class_weight> ::= balanced | None
    <class_weight_rf> ::= balanced | balanced_subsample | None
    <criterion_gb> ::= friedman_mse | squared_error
    <loss> ::= log_loss | exponential
    <max_leaves> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10
    """

    # Load grammar
    grammar = BNFGrammar()
    grammar.load_grammar(grammar_text)

    # Run GGP
    ggp = GrammarBasedGP(grammar)
    best_program = ggp.evolve()

    # Print the best program
    #print("Best Program Found:", )
