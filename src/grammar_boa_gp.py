import math
from copy import deepcopy
import time
import multiprocessing
import pandas as pd
import numpy as np
import warnings
import random
import fcntl
import os
from datetime import datetime
import pandas as pd

from .workflow.scaler_tranformer import ScalingTransformer
from .workflow.feature_selection_transformer import FeatureSelectionTransformer
from .workflow.ml_algorithm_transformer import MLAlgorithmTransformer

from sklearn.metrics import make_scorer, matthews_corrcoef, roc_auc_score, recall_score, average_precision_score, precision_score, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold


from pyAgrum.skbn import BNClassifier
import pyAgrum.lib.image as gumimage
import pyAgrum.skbn._MBCalcul as mbcalcul

import pyAgrum as gum

warnings.filterwarnings("ignore")




        



        


class GrammarBayesOptGeneticfProgAlgorithm:
    def __init__(self, grammar, training_dir, testing_dir, fitness_cache={}, num_cores=20, time_budget_minutes_alg_eval = 1, 
                 population_size=100, max_generations=10, max_time=60, mutation_rate=0.15, crossover_rate=0.8, 
                 crossover_mutation_rate=0.05, elitism_size=1, fitness_metric="mcc", 
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
                f.write(str(f) + "\n"  + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)  # Unlock the file              
            fitness_value = fitness_value * 0.7  # Set fitness value to zero if time budget exceeded
       
        if math.isnan(fitness_value):    
            return 0.0
        else:
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

    def convert_pipeline_to_df(self, pipeline, pipeline_string):
        """
        Transforms a single AutoML pipeline into a binary vector indicating the presence (1) 
        or absence (0) of a feature definition, scaler, selection method, or ML algorithm.
        """        
        pipeline_list = pipeline_string.split(" # ")
        main_buling_blocks = ""
        main_buling_blocks += pipeline_list[0] + " # "
        main_buling_blocks += pipeline_list[1].split(" ")[0] + " # "
        main_buling_blocks += pipeline_list[2].split(" ")[0] + " # "
        main_buling_blocks += pipeline_list[3].split(" ")[0]
        
        # Define all possible feature definition, scaling, selection, and ML algorithm options
        feature_definition_columns = [
            "General_Descriptors", "Advanced_Descriptors", "Graph_based_Signatures", "Toxicophores", "Fragments"
        ]
        
        scaling_columns = ["normalizer", "minmax_scaler", "maxabs_scaler","robust_scaler", "standard_scaler", "no_scaling"]
        selection_columns = [
            "variance_threshold", "select_percentile", "selectfpr", "selectfwe", "selectfdr", "select_rfe", "no_feature_selection"
        ]
        ml_algorithm_columns = [
            "neural_networks", "adaboost", "decision_tree", "extra_tree", "random_rorest",
            "extra_trees", "gradient_boosting", "xgboost", "svm", "nu_svm"
        ]
        
        # Combine all columns into a single list for the DataFrame
        all_columns = feature_definition_columns + scaling_columns + selection_columns + ml_algorithm_columns  

        row = {col: 0 for col in all_columns}  # Initialize all columns with 0
    
        for section in pipeline["<start>"]:
            if isinstance(section, dict):
                for key, value in section.items():
                    col_name = key.replace("<", "").replace(">", "")  # Normalize column names
    
                    # Check and set feature definitions
                    if col_name == "feature_definition":
                        for feature in value:
                            if feature in feature_definition_columns:
                                row[feature] = 1
    
                    # Check and set the first option for scaling, selection, and ML algorithms
                    elif col_name in ["feature_scaling", "feature_selection", "ml_algorithms"]:
                        method_name = list(value[0].keys())[0].replace("<", "").replace(">", "")
                        if method_name in scaling_columns + selection_columns + ml_algorithm_columns:
                            row[method_name] = 1
                            
        row["main_building_blocks"] = main_buling_blocks
        return row   

    def fit_BNC_and_get_MB(self, file_path, image_path):
        
        df = pd.read_csv(file_path, header=0, sep=",")

        #Define X and y
        X = df
        y = df["target"]
        X = X.drop("target", axis=1)
        X = X.drop("main_building_blocks", axis=1)
        X = X.drop("performance", axis=1)  

        #Define the Bayesian Network Classifier
        bnc = BNClassifier(learningMethod='GHC', scoringType='BIC')
        #And, fit it
        bnc.fit(X, y)

        # Get the Markov Blanket
        bn = bnc.bn
        mb = mbcalcul.compileMarkovBlanket(bn, "target")
        #gumimage.export(bn, "bnclook_" + image_path)
        gumimage.export(mb, image_path)
        mb.erase("target")   

        mb_list = list(mb)
        mb_building_blocks = []
        for bb in mb_list:
            mb_building_blocks.append(bb[1])
        return mb_building_blocks

    def sample_based_on_BNC(self, mb_building_blocks):
        max_sampling = int(self.population_size * 0.1)
        if(max_sampling == 0):
            max_sampling += 1
        count = 0

        feature_definitions = ["General_Descriptors", "Advanced_Descriptors", "Graph_based_Signatures", "Toxicophores", "Fragments"]
        scalings = ["Normalizer", "MinMaxScaler", "MaxAbsScaler","RobustScaler", "StandardScaler", "NoScaling"]
        feature_selections = ["VarianceThreshold", "SelectPercentile", "SelectFpr", "SelectFwe", "SelectFdr", "SelectRFE", "NoFeatureSelection"]
        
        ml_algorithms = ["NeuroNets", "AdaBoostClassifier", "DecisionTreeClassifier", "ExtraTreeClassifier", 
                         "RandomForestClassifier","ExtraTreesClassifier", "GradientBoostingClassifier", "XGBClassifier", 
                         "SVM", "NuSVM"]

        feature_definition_columns = [
            "General_Descriptors", "Advanced_Descriptors", "Graph_based_Signatures", "Toxicophores", "Fragments"
        ]
        
        scaling_columns = ["normalizer", "minmax_scaler", "maxabs_scaler","robust_scaler", "standard_scaler", "no_scaling"]
        
        selection_columns = [
            "variance_threshold", "select_percentile", "selectfpr", "selectfwe", "selectfdr", "select_rfe", "no_feature_selection"
        ]
        ml_algorithm_columns = [
            "neural_networks", "adaboost", "decision_tree", "extra_tree", "random_rorest",
            "extra_trees", "gradient_boosting", "xgboost", "svm", "nu_svm"
        ]

        scaling_dict = {"normalizer": "Normalizer", 
                           "minmax_scaler": "MinMaxScaler", 
                           "maxabs_scaler": "MaxAbsScaler",
                           "robust_scaler": "RobustScaler", 
                           "standard_scaler":"StandardScaler", 
                           "no_scaling": "NoScaling"}
        
        selection_dict = {"variance_threshold":"VarianceThreshold", 
                          "select_percentile":"SelectPercentile", 
                          "selectfpr":"SelectFpr", 
                          "selectfwe":"SelectFwe", 
                          "selectfdr":"SelectFdr", 
                          "select_rfe":"SelectRFE", 
                          "no_feature_selection":"NoFeatureSelection"}
        
        ml_algorithm_dict = {"neural_networks": "NeuroNets", 
                                "adaboost": "AdaBoostClassifier", 
                                "decision_tree": "DecisionTreeClassifier", 
                                "extra_tree": "ExtraTreeClassifier", 
                                "random_rorest": "RandomForestClassifier",
                                "extra_trees": "ExtraTreesClassifier", 
                                "gradient_boosting": "GradientBoostingClassifier", 
                                "xgboost": "XGBClassifier", 
                                "svm": "SVM", 
                                "nu_svm": "NuSVM"}

        feature_definitions_mb = []
        scalings_mb = []
        feature_selections_mb = []
        ml_algorithms_mb = []

        
        for building_block in mb_building_blocks:
            if building_block in feature_definition_columns:
                feature_definitions_mb.append(building_block)
            elif building_block in scaling_columns:
                scalings_mb.append(scaling_dict[building_block])
            elif building_block in selection_columns:
                feature_selections_mb.append(selection_dict[building_block])
            elif building_block in ml_algorithm_columns:
                ml_algorithms_mb.append(ml_algorithm_dict[building_block])                
                

        sampled_pipelines = []
        count_aux = 0
        while count < max_sampling:
            test_representation = False
            test_scaling = False
            test_feature_selection = False
            test_ml_algorithm = False
            
            trial = self.grammar.generate_parse_tree()
            trial_str = self.grammar.parse_tree_to_string(trial)
            
            trial_list = trial_str.split(" # ")            
            trial_rep = trial_list[0].split(" ")
            trial_scaling = trial_list[1].split(" ")[0]
            trial_feat_selection = trial_list[2].split(" ")[0]
            trial_ml_algorithm = trial_list[3].split(" ")[0]
           
            if feature_definitions_mb:
                for r in trial_rep:
                    if r in feature_definitions_mb:
                        test_representation = True
            else:
                test_representation = True

         
            if scalings_mb:
                if trial_scaling in scalings_mb:
                    test_scaling = True               
            else:
                test_scaling = True

            if feature_selections_mb:            
                if trial_feat_selection in feature_selections_mb:
                    test_feature_selection = True               
            else:
                test_feature_selection = True

           
            if ml_algorithms_mb:                  
                if trial_ml_algorithm in ml_algorithms_mb:
                    test_ml_algorithm = True               
            else:
                test_ml_algorithm = True                  
            
            if test_representation and test_scaling and test_feature_selection and test_ml_algorithm:
                count += 1
                sampled_pipelines.append(trial)
            count_aux+=1
            if(count_aux > 1500):
                break
          
            
        return sampled_pipelines

    def tournament_selection(self, population, fitness_scores, population_size, tournament_size=2):
        # Randomly select indices for the tournament
        indices = []

        while len(indices) < tournament_size:
            idx = random.randint(0, population_size - 1)
            indices.append(idx)

        # Get the index of the best individual among the selected
        best_idx = max(indices, key=lambda idx: fitness_scores[idx])  

        return population[best_idx]
        
        
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

        current_best = 0.0
        current_best_threshold = 0.0
        currewnt_worst_threshold = 0.0
        df_pipelines = pd.DataFrame()
        check_repeated = {}
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
            current_best = -1.00
            for p in self.fitness_cache:
                f = self.fitness_cache[p]
                if(f > current_best):
                    current_best = f
                
            
            current_best_threshold = current_best * 0.8
            current_worst_threshold = current_best * 0.6            
            
            elites = [self.population[i] for i in pop_indices[:self.elitism_size]] 
            # Elitism: retain the best individuals
            new_population = []
            new_population.extend(elites)

            #Recalculating threshold
            ind_count_pop = {}
            df_pipelines_aux = pd.DataFrame()
            
            if not df_pipelines.empty:
         
                # Group by 'main_building_blocks' and calculate the mean of 'performance'
                average_performance = df_pipelines.groupby('main_building_blocks')['performance'].mean().reset_index(name='average_performance')
                
                # Merge the average performance back into the original dataframe
                df_pipelines_new = df_pipelines.merge(average_performance, on='main_building_blocks')
                
                # If you want to keep just one sample per group (with all columns)
                df_pipelines_new.drop_duplicates(subset='main_building_blocks', keep='first', inplace=True)
                #remove previous performance column and rename the new one to performance
                df_pipelines_new = df_pipelines_new.drop("performance", axis=1)
                df_pipelines_new.rename(columns={'average_performance': 'performance'}, inplace=True)
                

                #Update the class based on current performance
                for i, row in df_pipelines.iterrows():
                    new_row = row                 
                    if new_row["performance"] >= current_best_threshold:
                        new_row["target"] = 1
                    elif new_row["performance"] <= current_worst_threshold:
                        new_row["target"] = 0
    
                    df_pipelines_aux = pd.concat([df_pipelines_aux, pd.DataFrame([new_row])], ignore_index=True)

            df_pipelines = pd.DataFrame()
            df_pipelines = df_pipelines_aux.copy(deep=True)
            list_cols =  list(df_pipelines.columns)
           
            
            #checking and creating performance class from individual's evaluation
            for i in pop_indices:
                ind = self.grammar.parse_tree_to_string(self.population[i])                
                new_row = self.convert_pipeline_to_df(self.population[i], ind)
                if ind not in check_repeated:
                    if fitness_scores[i] >= current_best_threshold:
                        new_row["target"] = 1
                        new_row["performance"] = fitness_scores[i]
                        df_pipelines = pd.concat([df_pipelines, pd.DataFrame([new_row])], ignore_index=True)
                        check_repeated[ind] = 1
                    elif fitness_scores[i] <= current_worst_threshold:
                        new_row["target"] = 0
                        new_row["performance"] = fitness_scores[i]
                        df_pipelines = pd.concat([df_pipelines, pd.DataFrame([new_row])], ignore_index=True)
                        check_repeated[ind] = 0
                    
                #df_pipelines.append()
                print(ind + "--->" + str(fitness_scores[i]))
                if(ind not in ind_count_pop):
                    ind_count_pop[ind] = 1
                else:
                    count = ind_count_pop[ind] + 1
                    ind_count_pop[ind] = count
            
            file_path = ""
            if not df_pipelines.empty:
                #list_cols =  list(df_pipelines.columns)
                file_path = "data_bnc" + str(generation)+".csv"
                if os.path.exists(file_path):  # Check if the file exists
                    os.remove(file_path)       # Delete the file
                df_pipelines.to_csv("data_bnc" + str(generation)+".csv", header=True, sep=",", index=False)
            
            image_path = "outputBNC_MB" + str(generation) + ".pdf"
            #Fitting and getting the Markov Blanket from the BNC that is causing the performance
            mb_building_blocks = self.fit_BNC_and_get_MB(file_path, image_path)  

            final_file_name_ab = self.experiment_name + "_BNC_markov_blanket.txt"
            final_result = ""
            with open(final_file_name_ab, "a") as file:
                final_result += "GENERATION " + str(generation) + ": " + str(mb_building_blocks)  + "\n"
                file.write(final_result)
                file.close()            
            #Sampling new pipelines in accordance to the Markov Blanket of the class node
            sampled_pipelines = self.sample_based_on_BNC(mb_building_blocks)
            #Extending the new population with the sampled pipelines from the BNC's Markov Blanket:
            new_population.extend(sampled_pipelines)

            
            #Adding new individuals if the population has >70% of the individuals are the same
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
                
                parent1 = self.tournament_selection(self.population, fitness_scores, self.population_size)
                parent2 = self.tournament_selection(self.population, fitness_scores, self.population_size)

                while parent1 == parent2:
                    parent2 = self.tournament_selection(self.population, fitness_scores, self.population_size)

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

