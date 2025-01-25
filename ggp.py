import random
from collections import defaultdict


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

    def generate_parse_tree(self, symbol: str = "<start>") -> dict:
        """
        Generates a parse tree starting from the given symbol.
        """
        if symbol not in self.grammar:
            # Terminal symbol (leaf node)
            return symbol
        # Select a random production
        production = random.choice(self.grammar[symbol])
        # Recursively generate parse tree for each part of the production
        return {symbol: [self.generate_parse_tree(token) for token in production]}

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
        Validates if the parse tree conforms to the grammar.
        """
        if isinstance(tree, str):
            # Terminal symbol
            return tree in self.terminals
        if not isinstance(tree, dict) or len(tree) != 1:
            return False

        root, children = list(tree.items())[0]
        if root != symbol:
            return False

        # Check if the children match any valid production
        for production in self.grammar[symbol]:
            if len(production) == len(children) and all(
                self.validate_parse_tree(child, production[i])
                for i, child in enumerate(children)
            ):
                return True
        return False


class GrammarBasedGP:
    def __init__(self, grammar, population_size=10, max_generations=20, mutation_rate=0.1, crossover_rate=0.7, elitism_size=2):
        self.grammar = grammar
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
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

    def mutate(self, individual):
        """
        Mutates an individual by randomly replacing a subtree.
        """
        if isinstance(individual, str):  # Terminal, no mutation possible
            return individual
        root, children = list(individual.items())[0]
        idx = random.randint(0, len(children) - 1)
        # Replace the selected subtree
        children[idx] = self.grammar.generate_parse_tree(root)
        # Validate mutation
        return individual if self.grammar.validate_parse_tree(individual) else individual

    def crossover(self, parent1, parent2):
        """
        Performs crossover by swapping subtrees between parents.
        """
        if isinstance(parent1, str) or isinstance(parent2, str):  # No crossover if terminal
            return parent1, parent2
        root1, children1 = list(parent1.items())[0]
        root2, children2 = list(parent2.items())[0]

        idx1 = random.randint(0, len(children1) - 1)
        idx2 = random.randint(0, len(children2) - 1)

        # Swap subtrees
        children1[idx1], children2[idx2] = children2[idx2], children1[idx1]

        # Validate offspring
        child1_valid = self.grammar.validate_parse_tree(parent1)
        child2_valid = self.grammar.validate_parse_tree(parent2)

        return (parent1 if child1_valid else parent1, parent2 if child2_valid else parent2)

    def evolve(self):
        """
        Runs the genetic programming algorithm.
        """
        # Initialize population
        self.population = [self.grammar.generate_parse_tree() for _ in range(self.population_size)]

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
                if random.random() < self.crossover_rate:
                    # Perform crossover
                    parent1, parent2 = random.choices(self.population, probabilities, k=2)
                    child1, child2 = self.crossover(deepcopy(parent1), deepcopy(parent2))
                    new_population.extend([child1, child2])
                else:
                    # Perform mutation
                    parent = random.choices(self.population, probabilities, k=1)[0]
                    child = self.mutate(deepcopy(parent))
                    new_population.append(child)

            # Trim excess individuals
            self.population = new_population[:self.population_size]

            # Print best individual of the generation
            best_fitness, best_individual = fitness_scores[0]
            print(f"Generation {generation}: Best Fitness = {best_fitness}")
            print(f"Best Individual: {self.grammar.parse_tree_to_string(best_individual)}")

        return fitness_scores[0][1]  # Return the best individual


# Example Usage
if __name__ == "__main__":
    random.seed(42)  # For reproducibility

    # Define grammar
    grammar_text = """
    <start> ::= <expr>
    <expr> ::= <expr> + <term> | <expr> - <term> | <term>
    <term> ::= <term> * <factor> | <term> / <factor> | <factor>
    <factor> ::= ( <expr> ) | <number>
    <number> ::= 1 | 2 | 3 | 4 | 5
    """

    # Load grammar
    grammar = BNFGrammar()
    grammar.load_grammar(grammar_text)

    # Run GGP
    ggp = GrammarBasedGP(grammar)
    best_program = ggp.evolve()

    # Print the best program
    print("Best Program Found:", grammar.parse_tree_to_string(best_program))
