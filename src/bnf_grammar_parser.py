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
