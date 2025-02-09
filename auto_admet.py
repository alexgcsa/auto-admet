import random
from collections import defaultdict
from copy import deepcopy
import time
import multiprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFpr, SelectFwe, SelectFdr, chi2, f_classif, RFE
import warnings
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")
from sklearn.metrics import make_scorer, matthews_corrcoef, roc_auc_score, recall_score, average_precision_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from datetime import datetime
import argparse
import fcntl

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
        

class MLAlgorithmTransformer:
    def __init__(self):
        pass

    def XGBoost(self, n_estimators_str, max_depth_str, max_leaves_str, learning_rate_str):
        max_depth_actual = None
        if max_depth_str != "None":
            max_depth_actual = int(max_depth_str)
    
        clf = XGBClassifier(n_estimators=int(n_estimators_str), max_depth=max_depth_actual, random_state=42, 
                            max_leaves=int(max_leaves_str), learning_rate=float(learning_rate_str), n_jobs=1)        
    
        return clf 
    
    
    def GradientBoosting(self, n_estimators_str, criterion_str, max_depth_str, min_samples_split_str, min_samples_leaf_str, max_features_str, loss_str):
        max_depth_actual = None
        if max_depth_str != "None":
            max_depth_actual = int(max_depth_str)
    
        max_features_actual = None
        if max_features_str != "None":
            max_features_actual = max_features_str  
    

        clf = GradientBoostingClassifier(n_estimators=int(n_estimators_str), criterion=criterion_str, max_depth=max_depth_actual, random_state=42, 
                                         min_samples_split=int(min_samples_split_str), min_samples_leaf=int(min_samples_split_str), 
                                         max_features=max_features_actual, loss=loss_str)        
    
        return clf      
 
    
    def ExtraTrees(self, n_estimators_str, criterion_str, max_depth_str, min_samples_split_str, min_samples_leaf_str, max_features_str, class_weight_str):
        max_depth_actual = None
        if max_depth_str != "None":
            max_depth_actual = int(max_depth_str)
    
        max_features_actual = None
        if max_features_str != "None":
            max_features_actual = max_features_str  
    
        class_weight_actual = None
        if class_weight_str != "None":
            class_weight_actual = class_weight_str   
     
            
        clf = ExtraTreesClassifier(n_estimators=int(n_estimators_str), criterion=criterion_str, max_depth=max_depth_actual, n_jobs=1, random_state=42, 
                                   class_weight=class_weight_actual,  min_samples_split=int(min_samples_split_str), 
                                   min_samples_leaf=int(min_samples_split_str), max_features=max_features_actual)        
    
        return clf  
    
    
    def RandomForest(self, n_estimators_str, criterion_str, max_depth_str, min_samples_split_str, min_samples_leaf_str, max_features_str, class_weight_str):
        max_depth_actual = None
        if max_depth_str != "None":
            max_depth_actual = int(max_depth_str)
    
        max_features_actual = None
        if max_features_str != "None":
            max_features_actual = max_features_str  
    
        class_weight_actual = None
        if class_weight_str != "None":
            class_weight_actual = class_weight_str   
     
        clf = RandomForestClassifier(n_estimators=int(n_estimators_str), criterion=criterion_str, max_depth=max_depth_actual, n_jobs=1, random_state=42, 
                                     class_weight=class_weight_actual,  min_samples_split=int(min_samples_split_str), 
                                     min_samples_leaf=int(min_samples_split_str), max_features=max_features_actual)        
 
        return clf   
        
    
    def ExtraTree(self, criterion_str, splitter_str, max_depth_str, min_samples_split_str, min_samples_leaf_str, max_features_str, class_weight_str):
        max_depth_actual = None
        if max_depth_str != "None":
            max_depth_actual = int(max_depth_str)
    
        max_features_actual = None
        if max_features_str != "None":
            max_features_actual = max_features_str  
    
        class_weight_actual = None
        if class_weight_str != "None":
            class_weight_actual = class_weight_str   
            
        clf = ExtraTreeClassifier(criterion=criterion_str, splitter='best', max_depth=max_depth_actual, 
                                  min_samples_split=int(min_samples_split_str), min_samples_leaf=int(min_samples_split_str),                                      
                                  max_features=max_features_actual, random_state=0)      
    
        return clf  
            
    
    def DecisionTree(self, criterion_str, splitter_str, max_depth_str, min_samples_split_str, min_samples_leaf_str, max_features_str, class_weight_str):
        max_depth_actual = None
        if max_depth_str != "None":
            max_depth_actual = int(max_depth_str)
    
        max_features_actual = None
        if max_features_str != "None":
            max_features_actual = max_features_str  
    
        class_weight_actual = None
        if class_weight_str != "None":
            class_weight_actual = class_weight_str   


        clf = DecisionTreeClassifier(criterion=criterion_str, splitter=splitter_str, max_depth=max_depth_actual, 
                                     min_samples_split=int(min_samples_split_str), min_samples_leaf=int(min_samples_split_str), 
                                     max_features=max_features_actual, random_state=0,
                                     class_weight=class_weight_actual)      
    
        return clf
    
    
    def AdaBoost(self, alg, n_est, lr):
        clf = AdaBoostClassifier(n_estimators=n_est, learning_rate=lr, algorithm=alg, random_state=0)
        return clf

    def SVM(self, kernel, degree, tol, max_iter, class_weight):
  
        #<class_weight> ::= balanced | None 
        actual_class_weight = None
        if(class_weight == "balanced"):
            actual_class_weight = "balanced"
            
        clf = SVC(kernel=str(kernel), degree=int(degree), probability=True, tol=float(tol), class_weight=actual_class_weight, max_iter=int(max_iter), random_state=0)
       
        return clf

    def NuSVM(self, kernel, degree, tol, max_iter, class_weight):
  
        #<class_weight> ::= balanced | None 
        actual_class_weight = None
        if(class_weight == "balanced"):
            actual_class_weight = "balanced"
            
        clf = NuSVC(kernel=str(kernel), degree=int(degree), probability=True, tol=float(tol), class_weight=actual_class_weight, max_iter=int(max_iter), random_state=0)
       
        return clf


    def NeuroNets(self, ml_algorithm_options):
        if(len(ml_algorithm_options)==6):
            hls = (int(ml_algorithm_options[0]),)
            af = ml_algorithm_options[1]
            sol = ml_algorithm_options[2]
            lr = ml_algorithm_options[3]
            mi = int(ml_algorithm_options[4])
            t = float(ml_algorithm_options[5])
        elif(len(ml_algorithm_options)==7):
            hls = (int(ml_algorithm_options[0]), ml_algorithm_options[1])
            af = ml_algorithm_options[2]
            sol = ml_algorithm_options[3]
            lr = ml_algorithm_options[4]
            mi = int(ml_algorithm_options[5])
            t = float(ml_algorithm_options[6])            
        elif(len(ml_algorithm_options)==8):
            hls = (int(ml_algorithm_options[0]), ml_algorithm_options[1], ml_algorithm_options[2])
            af = ml_algorithm_options[3]
            sol = ml_algorithm_options[4]
            lr = ml_algorithm_options[5]
            mi = int(ml_algorithm_options[6])
            t = float(ml_algorithm_options[7]) 
        
        clf = MLPClassifier(hidden_layer_sizes=hls, activation=af, solver=sol, learning_rate=lr, 
                          max_iter=mi, random_state=0, tol=t, early_stopping=True)

        return clf


class FeatureSelectionTransformer:
    def __init__(self, training_df, testing_df, training_label_col, error_log):
        self.training_df = training_df
        self.testing_df = testing_df
        self.training_label_col = training_label_col
        self.error_log = error_log
        self.model = None   

    def select_ml_algorithms(self, ml_algorithm):
        ml_alg_selection = MLAlgorithmTransformer()
        if ml_algorithm[0] == "AdaBoostClassifier":
            return ml_alg_selection.AdaBoost(str(ml_algorithm[1]), int(ml_algorithm[2]), float(ml_algorithm[3]))
        elif ml_algorithm[0] == "DecisionTreeClassifier":
            return ml_alg_selection.DecisionTree(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])    
        elif ml_algorithm[0] == "ExtraTreeClassifier":
            return ml_alg_selection.ExtraTree(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
        elif ml_algorithm[0] == "RandomForestClassifier":
            return ml_alg_selection.RandomForest(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
        elif ml_algorithm[0] == "ExtraTreesClassifier":
            return ml_alg_selection.ExtraTrees(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
        elif ml_algorithm[0] == "GradientBoostingClassifier":
            return ml_alg_selection.GradientBoosting(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7]) 
        elif ml_algorithm[0] == "XGBClassifier":
            return ml_alg_selection.XGBoost(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4])         
        elif ml_algorithm[0] == "SVM":
            return ml_alg_selection.SVM(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5]) 
        elif ml_algorithm[0] == "NuSVM":
            return ml_alg_selection.SVM(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5])             
        elif ml_algorithm[0] == "NeuroNets":
            return ml_alg_selection.NeuroNets(ml_algorithm[1:])             
                                              
        else:
            return None        

    def select_fwe(self, alpha_str, score_function_str):
    
        score_function_actual = f_classif
    
        if(score_function_str == "chi2"):
            score_function_actual = chi2       
        
        try:
            self.model = SelectFwe(score_func=score_function_actual, alpha = float(alpha_str)).fit(self.training_df, self.training_label_col)
            #df_np = self.model.transform(self.training_df)
    
            cols_idxs = self.model.get_support(indices=True)
            features_df_new = self.training_df.iloc[:,cols_idxs]
        
            return features_df_new
        except Exception as e:
            with open(self.error_log, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature selection - select_fwe" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file         
            return None 
            
    
    def select_fdr(self, alpha_str, score_function_str):
    
        score_function_actual = f_classif
    
        if(score_function_str == "chi2"):
            score_function_actual = chi2       
        
        try:
            self.model = SelectFdr(score_func=score_function_actual, alpha = float(alpha_str)).fit(self.training_df, self.training_label_col)
            #df_np = self.model.transform(self.training_df)
    
            cols_idxs = self.model.get_support(indices=True)
            features_df_new = self.training_df.iloc[:,cols_idxs]
        
            return features_df_new
        except Exception as e:
            with open(self.error_log, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature selection - select_fdr" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file        
            return None
            
    
    def select_fpr(self, alpha_str, score_function_str):    
        score_function_actual = f_classif    
        if(score_function_str == "chi2"):
            score_function_actual = chi2      
        
        try:
            self.model = SelectFpr(score_func=score_function_actual, alpha = float(alpha_str)).fit(self.training_df, self.training_label_col)      
            cols_idxs = self.model.get_support(indices=True)
            features_df_new = self.training_df.iloc[:,cols_idxs]
        
            return features_df_new
        except Exception as e:
            with open(self.error_log, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature selection - select_fpr" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file           
            return None
            
    
    def select_percentile(self, percentile_str, score_function_str):
        score_function_actual = f_classif    
        if(score_function_str == "chi2"):
            score_function_actual = chi2       
            
        try:
            self.model = SelectPercentile(score_func=score_function_actual, percentile = int(percentile_str)).fit(self.training_df, self.training_label_col)       
            cols_idxs = self.model.get_support(indices=True)
            features_df_new = self.training_df.iloc[:,cols_idxs]
        
            return features_df_new
        except Exception as e:
            with open(self.error_log, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature selection - select_percentile" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file           
            return None
            
    
    def variance_threshold(self,thrsh):
        try:
            self.model =VarianceThreshold(threshold=thrsh).fit(self.training_df, self.training_label_col)      
            cols_idxs = self.model.get_support(indices=True)
            features_df_new = self.training_df.iloc[:,cols_idxs]
        
            return features_df_new
        except Exception as e:            
            with open(self.error_log, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature selection - variance_threshold" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file 
            return None

    def select_rfe(self, n_features_to_select_rfe, step_rfe, ml_algorithm):
        try:
            
            estimator = self.select_ml_algorithms(ml_algorithm) 
            self.model = RFE(estimator, n_features_to_select=float(n_features_to_select_rfe), step=float(step_rfe)).fit(self.training_df, self.training_label_col)      
                           
            cols_idxs = self.model.get_support(indices=True)
            features_df_new = self.training_df.iloc[:,cols_idxs]
        
            return features_df_new
        except Exception as e:
            with open(self.error_log, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature selection - variance_threshold" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file 
            return None    

    def apply_model(self):
        try:
            self.testing_df = self.testing_df[self.training_df.columns]
            cols_idxs = self.model.get_support(indices=True)
            features_df_new_testing = self.testing_df.iloc[:,cols_idxs]
            df_np_testing = pd.DataFrame(self.model.transform(self.testing_df), columns=features_df_new_testing.columns)
            return features_df_new_testing
        except Exception as e:
            with open(self.error_log, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature selection - apply model" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file 
            return None
        


class ScalingTransformer:
    def __init__(self, training_df, testing_df, error_log):
        self.training_df = training_df
        self.testing_df = testing_df
        self.error_log = error_log
        self.model = None

    def normalizer(self, norm_hp):
        try:
            self.model = Normalizer(norm=norm_hp).fit(self.training_df)
            df_np = self.model.transform(self.training_df)
    
            return pd.DataFrame(df_np, columns = self.training_df.columns)
        except Exception as e:
            with open(self.error_log, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature scaling - scaling normalizer" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file           
            return None

    
    def max_abs_scaler(self):
        try:
            self.model = MaxAbsScaler().fit(self.training_df)
            df_np = self.model.transform(self.training_df)
    
            return pd.DataFrame(df_np, columns = self.training_df.columns)
        except Exception as e:
            with open(self.error_log, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature scaling - max_abs_scaler" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file            
            return None

    
    def min_max_scaler(self):
        try:
            self.model = MinMaxScaler().fit(self.training_df)
            df_np = self.model.transform(self.training_df)
    
            return pd.DataFrame(df_np, columns = self.training_df.columns)
        except Exception as e:
            with open(self.error_log, "a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature scaling - min_max_scaler" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file         
            return None 

    
    def standard_scaler(self, with_mean_str, with_std_str):
        with_mean_actual = True
        with_std_actual = True
    
        if with_mean_str == "False":
            with_mean_actual = False
        if with_std_str == "False":
            with_std_actual = False        
        try:
            self.model = StandardScaler(with_mean=with_mean_actual, with_std=with_std_actual).fit(self.training_df)
            df_np = self.model.transform(self.training_df)    
            return pd.DataFrame(df_np, columns = self.training_df.columns)
        except Exception as e:
            with open(self.error_log, "a") as f:            
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature scaling - standard_scaler" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file
            return None

    
    def robust_scaler(self, with_centering_str, with_scaling_str):
        with_centering_actual = True
        with_scaling_actual = True
    
        if with_centering_str == "False":
            with_centering_actual = False
        if with_scaling_str == "False":
            with_scaling_actual = False        
        try:
            self.model = RobustScaler(with_centering=with_centering_actual, with_scaling=with_scaling_actual).fit(self.training_df)
            df_np = self.model.transform(self.training_df)
    
            return pd.DataFrame(df_np, columns = self.training_df.columns)
        except Exception as e:
            with open(self.error_log, "a") as f:            
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature scaling - robust_scaler" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file            
            return None

    def apply_model(self):
        try:
            df_np_testing = pd.DataFrame(self.model.transform(self.testing_df), columns = self.testing_df.columns)
            return df_np_testing
        except Exception as e:
            with open(self.error_log, "a") as f:            
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature scaling - apply model" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file 
            return None
        


class GrammarBasedGP:
    def __init__(self, grammar, training_dir, testing_dir, fitness_cache={}, num_cores=20, time_budget_minutes_alg_eval = 3, 
                 population_size=20, max_generations=3, max_time=5, mutation_rate=0.15, crossover_rate=0.8, 
                 crossover_mutation_rate=0.05, elitism_size=1, fitness_metric="auc", 
                 experiment_name = "expABC", stopping_criterion = "time", seed=0):
        self.grammar = grammar
        self.training_dir = training_dir
        self.testing_dir = testing_dir
        self.fitness_cache = fitness_cache
        self.num_cores = num_cores
        self.time_budget_minutes_alg_eval = time_budget_minutes_alg_eval
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_time = max_time
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.crossover_mutation_rate = crossover_mutation_rate
        self.elitism_size = elitism_size
        self.fitness_metric = fitness_metric
        self.experiment_name = experiment_name
        self.stopping_criterion = stopping_criterion
        self.seed = seed        
        self.population = []

    def select_ml_algorithms(self, ml_algorithm):        
        ml_alg_selection = MLAlgorithmTransformer()
        if ml_algorithm[0] == "AdaBoostClassifier":
            return ml_alg_selection.AdaBoost(str(ml_algorithm[1]), int(ml_algorithm[2]), float(ml_algorithm[3]))
        elif ml_algorithm[0] == "DecisionTreeClassifier":
            return ml_alg_selection.DecisionTree(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])    
        elif ml_algorithm[0] == "ExtraTreeClassifier":
            return ml_alg_selection.ExtraTree(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
        elif ml_algorithm[0] == "RandomForestClassifier":
            return ml_alg_selection.RandomForest(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
        elif ml_algorithm[0] == "ExtraTreesClassifier":
            return ml_alg_selection.ExtraTrees(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
        elif ml_algorithm[0] == "GradientBoostingClassifier":
            return ml_alg_selection.GradientBoosting(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7]) 
        elif ml_algorithm[0] == "XGBClassifier":
            return ml_alg_selection.XGBoost(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4])  
        elif ml_algorithm[0] == "SVM":
            return ml_alg_selection.SVM(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5])             
        elif ml_algorithm[0] == "NuSVM":
            return ml_alg_selection.NuSVM(ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5])             
        elif ml_algorithm[0] == "NeuroNets":
            return ml_alg_selection.NeuroNets(ml_algorithm[1:])             
                                                                  
        else:
            return None    

    
    def select_features(self, feature_selection, ml_algorithm, training_dataset_df, training_label_col, testing_dataset_df=None, testing=False):

        cp_training_dataset_df = training_dataset_df.copy(deep=True)
        cp_testing_datset_df = None        
        if(testing):
            cp_testing_datset_df = testing_dataset_df.copy(deep=True)

        cp_training_label_col = training_label_col.copy(deep=True)
        error_log = self.experiment_name + "_error.log"
        feature_selection_transformer = FeatureSelectionTransformer(cp_training_dataset_df, cp_testing_datset_df, cp_training_label_col, error_log)
        mod_training_dataset_df = None
        mod_testing_dataset_df = None        
        if feature_selection[0] == "NoFeatureSelection":
            if(testing):
                return training_dataset_df, testing_dataset_df
            else:
                return training_dataset_df
        elif feature_selection[0] == "VarianceThreshold":
            mod_training_dataset_df = feature_selection_transformer.variance_threshold(float(feature_selection[1]))
            if(testing):
                mod_testing_dataset_df = feature_selection_transformer.apply_model()
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df

        elif feature_selection[0] == "SelectPercentile":        
            mod_training_dataset_df = feature_selection_transformer.select_percentile(feature_selection[1], feature_selection[2])    
            if(testing):
                mod_testing_dataset_df = feature_selection_transformer.apply_model()
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df   
        elif feature_selection[0] == "SelectFpr":        
            mod_training_dataset_df = feature_selection_transformer.select_fpr(feature_selection[1], feature_selection[2])    
            if(testing):
                mod_testing_dataset_df = feature_selection_transformer.apply_model()
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df  
        elif feature_selection[0] == "SelectFdr":        
            mod_training_dataset_df = feature_selection_transformer.select_fdr(feature_selection[1], feature_selection[2])    
            if(testing):
                mod_testing_dataset_df = feature_selection_transformer.apply_model()
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df  
        elif feature_selection[0] == "SelectFwe":        
            mod_training_dataset_df = feature_selection_transformer.select_fwe(feature_selection[1], feature_selection[2])    
            if(testing):
                mod_testing_dataset_df = feature_selection_transformer.apply_model()
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df  
        elif feature_selection[0] == "SelectRFE":        
            mod_training_dataset_df = feature_selection_transformer.select_rfe(feature_selection[1], feature_selection[2], ml_algorithm)    
            if(testing):
                mod_testing_dataset_df = feature_selection_transformer.apply_model()               
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df                  
        else:
            return None       

    
    def scale_features(self, feature_scaling, training_dataset_df, testing_dataset_df=None, testing=False):

        cp_training_dataset_df = training_dataset_df.copy(deep=True)
        cp_testing_datset_df = None
        if(testing):
            cp_testing_datset_df = testing_dataset_df.copy(deep=True)
        error_log = self.experiment_name + "_error.log"
        scaling_transformer = ScalingTransformer(cp_training_dataset_df, cp_testing_datset_df, error_log)
        mod_training_dataset_df = None
        mod_testing_dataset_df = None
        if feature_scaling[0] == "NoScaling":
            if(testing):
                return training_dataset_df, testing_dataset_df
            else:
                return training_dataset_df
            
        elif feature_scaling[0] == "Normalizer":
            mod_training_dataset_df = scaling_transformer.normalizer(str(feature_scaling[1]))
            if(testing):
                mod_testing_dataset_df = scaling_transformer.apply_model()
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df
        elif feature_scaling[0] == "MinMaxScaler":
            mod_training_dataset_df = scaling_transformer.min_max_scaler()
            if(testing):
                mod_testing_dataset_df = scaling_transformer.apply_model()
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df
        elif feature_scaling[0] == "MaxAbsScaler":
            mod_training_dataset_df = scaling_transformer.max_abs_scaler()
            if(testing):
                mod_testing_dataset_df = scaling_transformer.apply_model()
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df  
        elif feature_scaling[0] == "StandardScaler":
            mod_training_dataset_df  = scaling_transformer.standard_scaler(feature_scaling[1], feature_scaling[2])
            if(testing):
                mod_testing_dataset_df = scaling_transformer.apply_model()
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df
        elif feature_scaling[0] == "RobustScaler":
            mod_training_dataset_df = scaling_transformer.robust_scaler(feature_scaling[1], feature_scaling[2])
            if(testing):
                mod_testing_dataset_df = scaling_transformer.apply_model()
                return mod_training_dataset_df, mod_testing_dataset_df
            else:
                return mod_training_dataset_df      
        else:            
            return None    
    


    def represent_molecules(self, list_of_feature_types, training_dataset_df, testing_dataset_df=None, testing=False):
        """
        represents a chemical dataset with descriptors.
        """          
    
        columns = []
        for lft in list_of_feature_types:
            if lft == "General_Descriptors":
                columns += ["HeavyAtomCount","MolLogP","NumHeteroatoms","NumRotatableBonds","RingCount","TPSA","LabuteASA","MolWt","FCount","FCount2","Acceptor_Count","Aromatic_Count","Donor_Count","Hydrophobe_Count","NegIonizable_Count","PosIonizable_Count",]
            elif lft == "Advanced_Descriptors":
                columns += ["BalabanJ","BertzCT","Chi0","Chi0n","Chi0v","Chi1","Chi1n","Chi1v","Chi2n","Chi2v","Chi3n","Chi3v","Chi4n","Chi4v","HallKierAlpha","Kappa1","Kappa2","Kappa3","NHOHCount","NOCount","PEOE_VSA1","PEOE_VSA10","PEOE_VSA11","PEOE_VSA12","PEOE_VSA13","PEOE_VSA14","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA5","PEOE_VSA6","PEOE_VSA7","PEOE_VSA8","PEOE_VSA9","SMR_VSA1","SMR_VSA10","SMR_VSA2","SMR_VSA3","SMR_VSA4","SMR_VSA5","SMR_VSA6","SMR_VSA7","SMR_VSA8","SMR_VSA9","SlogP_VSA1","SlogP_VSA10","SlogP_VSA11","SlogP_VSA12","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA5","SlogP_VSA6","SlogP_VSA7","SlogP_VSA8","SlogP_VSA9","VSA_EState1","VSA_EState10","VSA_EState2","VSA_EState3","VSA_EState4","VSA_EState5","VSA_EState6","VSA_EState7","VSA_EState8","VSA_EState9"]
            elif lft == "Toxicophores":
                columns += ["Tox_1","Tox_2","Tox_3","Tox_4","Tox_5","Tox_6","Tox_7","Tox_8","Tox_9","Tox_10","Tox_11","Tox_12","Tox_13","Tox_14","Tox_15","Tox_16","Tox_17","Tox_18","Tox_19","Tox_20","Tox_21","Tox_22","Tox_23","Tox_24","Tox_25","Tox_26","Tox_27","Tox_28","Tox_29","Tox_30","Tox_31","Tox_32","Tox_33","Tox_34","Tox_35","Tox_36"]
            elif lft == "Fragments":
                columns += ["fr_Al_COO","fr_Al_OH","fr_Al_OH_noTert","fr_ArN","fr_Ar_COO","fr_Ar_N","fr_Ar_NH","fr_Ar_OH","fr_COO","fr_COO2","fr_C_O","fr_C_O_noCOO","fr_C_S","fr_HOCCN","fr_Imine","fr_NH0","fr_NH1","fr_NH2","fr_N_O","fr_Ndealkylation1","fr_Ndealkylation2","fr_Nhpyrrole","fr_SH","fr_aldehyde","fr_alkyl_carbamate","fr_alkyl_halide","fr_allylic_oxid","fr_amide","fr_amidine","fr_aniline","fr_aryl_methyl","fr_azide","fr_azo","fr_barbitur","fr_benzene","fr_benzodiazepine","fr_bicyclic","fr_diazo","fr_dihydropyridine","fr_epoxide","fr_ester","fr_ether","fr_furan","fr_guanido","fr_halogen","fr_hdrzine","fr_hdrzone","fr_imidazole","fr_imide","fr_isocyan","fr_isothiocyan","fr_ketone","fr_ketone_Topliss","fr_lactam","fr_lactone","fr_methoxy","fr_morpholine","fr_nitrile","fr_nitro","fr_nitro_arom","fr_nitro_arom_nonortho","fr_nitroso","fr_oxazole","fr_oxime","fr_para_hydroxylation","fr_phenol","fr_phenol_noOrthoHbond","fr_phos_acid","fr_phos_ester","fr_piperdine","fr_piperzine","fr_priamide","fr_prisulfonamd","fr_pyridine","fr_quatN","fr_sulfide","fr_sulfonamd","fr_sulfone","fr_term_acetylene","fr_tetrazole","fr_thiazole","fr_thiocyan","fr_thiophene","fr_unbrch_alkane","fr_urea"]
            elif lft == "Graph_based_Signatures":
                columns += ["Acceptor:Acceptor-6.00","Acceptor:Aromatic-6.00","Acceptor:Donor-6.00","Acceptor:Hydrophobe-6.00","Acceptor:NegIonizable-6.00","Acceptor:PosIonizable-6.00","Aromatic:Aromatic-6.00","Aromatic:Donor-6.00","Aromatic:Hydrophobe-6.00","Aromatic:NegIonizable-6.00","Aromatic:PosIonizable-6.00","Donor:Donor-6.00","Donor:Hydrophobe-6.00","Donor:NegIonizable-6.00","Donor:PosIonizable-6.00","Hydrophobe:Hydrophobe-6.00","Hydrophobe:NegIonizable-6.00","Hydrophobe:PosIonizable-6.00","NegIonizable:NegIonizable-6.00","NegIonizable:PosIonizable-6.00","PosIonizable:PosIonizable-6.00","Acceptor:Acceptor-4.00","Acceptor:Aromatic-4.00","Acceptor:Donor-4.00","Acceptor:Hydrophobe-4.00","Acceptor:NegIonizable-4.00","Acceptor:PosIonizable-4.00","Aromatic:Aromatic-4.00","Aromatic:Donor-4.00","Aromatic:Hydrophobe-4.00","Aromatic:NegIonizable-4.00","Aromatic:PosIonizable-4.00","Donor:Donor-4.00","Donor:Hydrophobe-4.00","Donor:NegIonizable-4.00","Donor:PosIonizable-4.00","Hydrophobe:Hydrophobe-4.00","Hydrophobe:NegIonizable-4.00","Hydrophobe:PosIonizable-4.00","NegIonizable:NegIonizable-4.00","NegIonizable:PosIonizable-4.00","PosIonizable:PosIonizable-4.00","Acceptor:Acceptor-2.00","Acceptor:Aromatic-2.00","Acceptor:Donor-2.00","Acceptor:Hydrophobe-2.00","Acceptor:NegIonizable-2.00","Acceptor:PosIonizable-2.00","Aromatic:Aromatic-2.00","Aromatic:Donor-2.00","Aromatic:Hydrophobe-2.00","Aromatic:NegIonizable-2.00","Aromatic:PosIonizable-2.00","Donor:Donor-2.00","Donor:Hydrophobe-2.00","Donor:NegIonizable-2.00","Donor:PosIonizable-2.00","Hydrophobe:Hydrophobe-2.00","Hydrophobe:NegIonizable-2.00","Hydrophobe:PosIonizable-2.00","NegIonizable:NegIonizable-2.00","NegIonizable:PosIonizable-2.00","PosIonizable:PosIonizable-2.00"]
            
        mod_training_dataset_df = None
        mod_testing_dataset_df = None
        try:
            cp_training_dataset_df = training_dataset_df.copy(deep=True)
            mod_training_dataset_df = cp_training_dataset_df[columns]

            if(testing):
                cp_testing_dataset_df = testing_dataset_df.copy(deep=True)
                mod_testing_dataset_df = cp_testing_dataset_df[columns]                
                
        except:
            error_log = self.experiment_name + "_error.log"
            with open(error_log, "a") as f:            
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on feature representation" + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file 
    
        if(testing):
            return mod_training_dataset_df, mod_testing_dataset_df
        else:
            return mod_training_dataset_df   



    def evaluate_train_test(self, pipeline):
        """
        Evaluates the pipeline on training and testing, performing each step of the ML pipeline.
        """  
        
        start_time = time.time()

        #all the steps in Auto-ADMET pipeline:
        pipeline_string = self.grammar.parse_tree_to_string(pipeline)
        pipeline_list = pipeline_string.split(" # ")
        representation = pipeline_list[0].split(" ")
        feature_scaling = pipeline_list[1].split(" ")
        feature_selection = pipeline_list[2].split(" ")
        ml_algorithm = pipeline_list[3].split(" ")

        #applying the steps to an actual dataset:
        training_dataset_df = pd.read_csv(self.training_dir, header=0, sep=",")
        training_label_col = training_dataset_df["CLASS"]
        training_dataset_df = training_dataset_df.drop("CLASS", axis=1)
        training_dataset_df = training_dataset_df.drop("ID", axis=1)
        training_dataset_df_cols = training_dataset_df.columns
        
        testing_dataset_df = pd.read_csv(self.testing_dir, header=0, sep=",")
        testing_label_col = testing_dataset_df["CLASS"]
        testing_dataset_df = testing_dataset_df.drop("CLASS", axis=1)
        testing_dataset_df = testing_dataset_df.drop("ID", axis=1)
        testing_dataset_df = testing_dataset_df[training_dataset_df_cols]

        rep_training_dataset_df, rep_testing_dataset_df = self.represent_molecules(representation, training_dataset_df, testing_dataset_df, True)
        prep_training_dataset_df, prep_testing_dataset_df = self.scale_features(feature_scaling, rep_training_dataset_df, rep_testing_dataset_df, True)
        sel_training_dataset_df, sel_testing_dataset_df = self.select_features(feature_selection, ml_algorithm, prep_training_dataset_df, training_label_col, testing_dataset_df, True)
        sel_testing_dataset_df = sel_testing_dataset_df[sel_training_dataset_df.columns]
        
        try:
            
            ml_algorithm  = self.select_ml_algorithms(ml_algorithm)
            ml_model = ml_algorithm.fit(sel_training_dataset_df, training_label_col)
            predictions = ml_model.predict(sel_testing_dataset_df)
            probabilities = ml_model.predict_proba(sel_testing_dataset_df)[:, 1]
            actuals = np.array(testing_label_col)
            
            mcc_test = round(matthews_corrcoef(actuals, predictions), 4)
            auc_test = round(roc_auc_score(actuals, probabilities), 4)
            rec_test = round(recall_score(actuals, predictions), 4)
            apr_test = round(average_precision_score(actuals, predictions), 4)
            prec_test = round(precision_score(actuals, predictions), 4)
            acc_test = round(accuracy_score(actuals, predictions), 4)
            return mcc_test, auc_test, rec_test, apr_test, prec_test, acc_test        
        except Exception as e:
            error_log = self.experiment_name + "_error.log"
            with open(error_log, "a") as f:            
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on pipeline - fitting" + "\n")
                f.write(pipeline_string + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file             
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    def evaluate_fitness(self, pipeline, dataset_path, time_budget_minutes_alg_eval):
        """
        evaluates pipeline with the fitness, performing each step of the ML pipeline.
        """  
        
        start_time = time.time()

        #all the steps in Auto-ADMET pipeline:
        pipeline_string = self.grammar.parse_tree_to_string(pipeline)
        pipeline_list = pipeline_string.split(" # ")
        representation = pipeline_list[0].split(" ")
        feature_scaling = pipeline_list[1].split(" ")
        feature_selection = pipeline_list[2].split(" ")
        ml_algorithm = pipeline_list[3].split(" ")

        #applying the steps to an actual dataset:
        dataset_df = pd.read_csv(self.training_dir, header=0, sep=",")
        label_col = dataset_df["CLASS"]
        dataset_df = dataset_df.drop("CLASS", axis=1)
        dataset_df = dataset_df.drop("ID", axis=1)

        rep_dataset_df = self.represent_molecules(representation, dataset_df)
        if(rep_dataset_df is None):
            return 0.0
     
        prep_dataset_df = self.scale_features(feature_scaling, rep_dataset_df)
        if(prep_dataset_df is None):
            return 0.0
            
        sel_dataset_df = self.select_features(feature_selection, ml_algorithm, prep_dataset_df, label_col)        
        if(sel_dataset_df is None):
            return 0.0

        ml_algorithm  = self.select_ml_algorithms(ml_algorithm)
        sel_dataset_df["CLASS"] = pd.Series(label_col)
        
        final_scores = []
        trials = range(3)
        for t in trials: 
            current_seed = self.seed + t
            outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=current_seed)
            try:
                y = sel_dataset_df.iloc[:,-1:]
                X = sel_dataset_df[sel_dataset_df.columns[:-1]]
                scores = None
                
                if(self.fitness_metric == "auc"):                
                    scores = cross_val_score(ml_algorithm, X, y, cv=outer_cv, scoring=make_scorer(roc_auc_score))
                elif(self.fitness_metric == "mcc"):            
                    scores = cross_val_score(ml_algorithm, X, y, cv=outer_cv, scoring=make_scorer(matthews_corrcoef))
                elif(self.fitness_metric == "recall"):
                    scores = cross_val_score(ml_algorithm, X, y, cv=outer_cv, scoring=make_scorer(recall_score))
                elif(self.fitness_metric == "precision"):
                    scores = cross_val_score(ml_algorithm, X, y, cv=outer_cv, scoring=make_scorer(precision_score))
                elif(self.fitness_metric == "auprc"):
                    scores = cross_val_score(ml_algorithm, X, y, cv=outer_cv, scoring=make_scorer(average_precision_score))
                elif(self.fitness_metric == "accuracy"):
                    scores = cross_val_score(ml_algorithm, X, y, cv=outer_cv, scoring=make_scorer(accuracy_score))                
    
                final_scores += list(scores)               
            except Exception as e:
                error_log = self.experiment_name + "_error.log"
                with open(error_log, "a") as f:            
                    fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                    f.write("Error on calculation scores - fitting" + "\n")
                    f.write(pipeline_string + "\n")
                    f.write(str(e) + "\n"  + "\n")
                    fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file 
                final_scores += [0.0, 0.0, 0.0]
                           
        
        # This function should evaluate the fitness of the individual within the time budget        
        elapsed_time = time.time() - start_time    
        fitness_value = np.array(final_scores).mean()
        if elapsed_time > (time_budget_minutes_alg_eval * 60):  # Check if elapsed time exceeds time budget
            error_log = self.experiment_name + "_error.log"
            with open(error_log, "a") as f:            
                fcntl.flock(f, fcntl.LOCK_EX)  # Lock the file
                f.write("Error on pipeline - exceeded time budget" + "\n")
                f.write(pipeline_string + "\n")
                f.write(str(e) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file              
            fitness_value = fitness_value * 0.7  # Set fitness value to zero if time budget exceeded
       
            
        return fitness_value

    
        
    def fitness(self):
        """
        Calculates the fitness function in parallel using multiprocessing,
        while caching results to avoid redundant evaluations.
        """
        with multiprocessing.Pool(processes=self.num_cores) as pool:
            results = []
            async_results = []
            
            # Submit all tasks asynchronously, checking cache first
            for pipeline in self.population:
                pipeline_str =self.grammar.parse_tree_to_string(pipeline)  # Convert individual to a string representation
                
                if pipeline_str in self.fitness_cache:
                    # Use cached value if available
                    results.append((pipeline, self.fitness_cache[pipeline_str]))
                else:
                    # Otherwise, evaluate it asynchronously
                    async_result = pool.apply_async(
                        self.evaluate_fitness, 
                        (pipeline, self.training_dir, self.time_budget_minutes_alg_eval)
                    )
                    async_results.append((pipeline, async_result))
    
            # Collect results in a non-blocking way
            for pipeline, async_result in async_results:
                try:
                    fitness_value = async_result.get(timeout=self.time_budget_minutes_alg_eval * 60)
                except multiprocessing.TimeoutError:
                    fitness_value = 0.0  # Timeout case
                
                # Cache the computed fitness value
                pipeline_str =self.grammar.parse_tree_to_string(pipeline)
               
                self.fitness_cache[pipeline_str] = fitness_value  # Store in dictionary
                results.append((pipeline, fitness_value))
        
        # Separate pipelines and fitness values
        pipelines, fitness_results = zip(*results) if results else ([], [])
    
        return list(pipelines), list(fitness_results)


    
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
            replacement = self.grammar.generate_parse_tree(components[idx // 2], max_depth=max_mutation_depth)
            children[idx] = replacement
        else:
            # Mutate other non-terminals
            idx = random.randint(0, len(children) - 1)
            children[idx] = self.grammar.generate_parse_tree(root, max_depth=max_mutation_depth)
    
        return individual
    
    
    def evolve(self):
        """
        Runs the genetic programming algorithm.
        """
        # Initialize population
        self.population = [self.grammar.generate_parse_tree() for _ in range(self.population_size)]
        pop_indices = []  
        
        generation = 0
        start = datetime.now()
        end = start
        time_diff_minutes = (end - start).total_seconds() / 60
        condition = ""
        if(self.stopping_criterion == "generations"):
            condition = generation < self.max_generations
        elif(self.stopping_criterion == "time"):
            condition = time_diff_minutes < (self.max_time - 0.5)

        #seed_index = 1
        #random.seed(seed_index)
        while condition:   
            print("GENERATION: " + str(generation))
            if(self.stopping_criterion == "generations"):
                condition = generation < self.max_generations
            elif(self.stopping_criterion == "time"):
                condition = time_diff_minutes < (self.max_time - 0.5)            
            
            #condition = generation < self.max_generations
            # Evaluate fitness
            pop_fitness_scores = self.fitness()
            evaluated_population = pop_fitness_scores[0]
            self.population = deepcopy(evaluated_population)
            fitness_scores = pop_fitness_scores[1]

            pop_indices = sorted(range(len(self.population)), key=lambda i: fitness_scores[i], reverse=True)
            elites = [self.population[i] for i in pop_indices[:self.elitism_size]] 
            # Elitism: retain the best individuals
            new_population = []
            new_population.extend(elites)

            #Adding new individuals if the population has >80% of the individuals are the same
            ind_count_pop = {}            
            for i in pop_indices:
                ind = self.grammar.parse_tree_to_string(self.population[i])
                print(ind + "--->" + str(fitness_scores[i]))
                if(ind not in ind_count_pop):
                    ind_count_pop[ind] = 1
                else:
                    count = ind_count_pop[ind] + 1
                    ind_count_pop[ind] = count

            max_count = -1
            for ind in ind_count_pop:
                ind_count = ind_count_pop[ind]
                if  ind_count > max_count:
                    max_count = ind_count

            population_stabilisation_rate = float(max_count)/float(self.population_size)
            
            if(population_stabilisation_rate > 0.7):
                new_ind1 = self.grammar.generate_parse_tree()
                new_ind2 = self.grammar.generate_parse_tree()
                new_population.append(new_ind1)
                new_population.append(new_ind2)

            # Selection probabilities
            fitness_values = [1.0 / (f + 1e-6) for f in fitness_scores]
            total_fitness = sum(fitness_values)
            probabilities = [f / total_fitness for f in fitness_values]

            while len(new_population) < self.population_size:
                
                idx1 = random.randint(0, len(self.population) - 1)
                idx2 = random.randint(0, len(self.population) - 1)
                while idx1 == idx2:
                    idx2 = random.randint(0, len(self.population) - 1)
                   
                parent1 = self.population[idx1]
                parent2 = self.population[idx2]

                random_num = random.random()
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
            end =  datetime.now()
            time_diff_minutes = (end - start).total_seconds() / 60
            generation += 1
            print("-----------------------------------------------")            

        best_indices = sorted(range(len(self.population)), key=lambda i: fitness_scores[i], reverse=True)[:1]
        best_fitness = [fitness_scores[i] for i in pop_indices][0]  
        best_individual = self.population[best_indices[0]]
        mcc, auc, rec, apr, prec, acc = self.evaluate_train_test(best_individual)
        
        final_file_name = self.experiment_name + ".txt"
        final_result = ""
        with open(final_file_name, "a") as file:
            final_result += self.experiment_name + ";"
            final_result += str(self.seed) + ";"
            final_result += str(generation) + ";"
            final_result += str(round(time_diff_minutes, 4)) + ";"
            final_result += str(mcc) + ";"
            final_result += str(auc) + ";"
            final_result += str(rec) + ";"
            final_result += str(apr) + ";"
            final_result += str(prec) + ";"
            final_result += str(acc) + ";" + ";"
            final_result += self.grammar.parse_tree_to_string(best_individual) + "\n"
            print(final_result)
            file.write(final_result)
            file.close()


# Example Usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")

    # Add arguments
    parser.add_argument("training_dir", type=str, help="Directory to the training dataset.")
    parser.add_argument("testing_dir", type=str, help="Directory to the training dataset.")
    parser.add_argument("grammar_dir", type=str, help="Directory to the grammar.")
    parser.add_argument("-s", "--seed", type=int, help="The seed", default=1) 
    parser.add_argument("-m", "--metric", type=str, help="The metric to be used during Auto-ADMET optimisation procedure", default="auc")
    parser.add_argument("-e", "--exp_name", type=str, help="The name of the experiment", default="Exp_ADMET")
    parser.add_argument("-t", "--time", type=int, help="Time in minutes to run the method", default=60)
    
    # Parse arguments
    args = parser.parse_args()
    training_dir = args.training_dir
    testing_dir = args.training_dir
    grammar_dir = args.grammar_dir
    seed = args.seed
    metric = args.metric
    exp_name = args.exp_name
    
    random.seed(seed)  # For reproducibility

    # Define grammar
    with open(grammar_dir, "r") as file:
        grammar_text = file.read()

    # Load grammar
    grammar = BNFGrammar()
    grammar.load_grammar(grammar_text)

    # Run GGP
    ggp = GrammarBasedGP(grammar, training_dir, testing_dir, fitness_metric=metric, experiment_name=exp_name)
    best_program = ggp.evolve()
