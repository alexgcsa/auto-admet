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
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import numpy as np
import random
import argparse

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

def crossover(parent1, parent2, grammar, seed):
    crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return ensure_valid_sentence(child1, grammar), ensure_valid_sentence(child2, grammar)

def ensure_valid_sentence(sentence, grammar):
    valid_symbols = set(symbol for expansion in grammar.values() for symbols in expansion for symbol in symbols)
    valid_symbols.add("#")
    valid_symbols.add("$")
    return "".join(symbol for symbol in sentence if symbol in valid_symbols)

def mutate(sentence, grammar, start_symbol, seed, mutation_rate=0.1):
    mutated_sentence = ""
    for symbol in sentence:
        if random.random() < mutation_rate:
            mutated_sentence += random.choice(grammar.get(symbol, [symbol]))
        else:
            mutated_sentence += symbol
    return mutated_sentence

def check_erratic_phenotypes(list_invalid, phenotype):
    for i in list_invalid:
        if i in phenotype:
            return True

    return False
        
    

def evolve(population, grammar, start_symbol, mutation_rate, crossover_rate, dataset_path, time_budget_minutes_alg_eval, num_cores, resample, generation, seed):
    pop_fitness_scores = evaluate_population_parallel(population, dataset_path, time_budget_minutes_alg_eval, num_cores, resample, generation)
    population = pop_fitness_scores[0]
    fitness_scores = pop_fitness_scores[1]
    total_fitness = sum(fitness_scores)
    new_population = []

    for i in range(len(population)):
        print("%s, %f"%(population[i], fitness_scores[i]))

        

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
        
        child1 = mutate(child1, grammar, start_symbol, seed, mutation_rate)
        child2 = mutate(child2, grammar, start_symbol, seed, mutation_rate)


        if ("$$" not in child1) and ("$#" not in child1) and ("$5" not in child1) and ("##" not in child1) and ("###$" not in child1):
            new_population.append(child1)

        if len(new_population) < len(population):
            if ("$$" not in child2) and ("$#" not in child2) and ("$5" not in child2) and ("##" not in child2) and ("###$" not in child2):
                new_population.append(child2)            

    return new_population

def represent(list_of_feature_types, dataset_df):    

    columns = []
    for lft in list_of_feature_types:
        if lft == "General_Descriptors":
            columns += ["HeavyAtomCount","MolLogP","NumHeteroatoms","NumRotatableBonds","RingCount","TPSA","LabuteASA","MolWt","FCount","FCount2"]
        elif lft == "Advanced_Descriptors":
            columns += ["BalabanJ","BertzCT","Chi0","Chi0n","Chi0v","Chi1","Chi1n","Chi1v","Chi2n","Chi2v","Chi3n","Chi3v","Chi4n","Chi4v","HallKierAlpha","Kappa1","Kappa2","Kappa3","NHOHCount","NOCount","PEOE_VSA1","PEOE_VSA10","PEOE_VSA11","PEOE_VSA12","PEOE_VSA13","PEOE_VSA14","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA5","PEOE_VSA6","PEOE_VSA7","PEOE_VSA8","PEOE_VSA9","SMR_VSA1","SMR_VSA10","SMR_VSA2","SMR_VSA3","SMR_VSA4","SMR_VSA5","SMR_VSA6","SMR_VSA7","SMR_VSA8","SMR_VSA9","SlogP_VSA1","SlogP_VSA10","SlogP_VSA11","SlogP_VSA12","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA5","SlogP_VSA6","SlogP_VSA7","SlogP_VSA8","SlogP_VSA9","VSA_EState1","VSA_EState10","VSA_EState2","VSA_EState3","VSA_EState4","VSA_EState5","VSA_EState6","VSA_EState7","VSA_EState8","VSA_EState9"]
        elif lft == "Toxicophores":
            columns += ["Tox_1","Tox_2","Tox_3","Tox_4","Tox_5","Tox_6","Tox_7","Tox_8","Tox_9","Tox_10","Tox_11","Tox_12","Tox_13","Tox_14","Tox_15","Tox_16","Tox_17","Tox_18","Tox_19","Tox_20","Tox_21","Tox_22","Tox_23","Tox_24","Tox_25","Tox_26","Tox_27","Tox_28","Tox_29","Tox_30","Tox_31","Tox_32","Tox_33","Tox_34","Tox_35","Tox_36"]
        elif lft == "Fragments":
            columns += ["fr_Al_COO","fr_Al_OH","fr_Al_OH_noTert","fr_ArN","fr_Ar_COO","fr_Ar_N","fr_Ar_NH","fr_Ar_OH","fr_COO","fr_COO2","fr_C_O","fr_C_O_noCOO","fr_C_S","fr_HOCCN","fr_Imine","fr_NH0","fr_NH1","fr_NH2","fr_N_O","fr_Ndealkylation1","fr_Ndealkylation2","fr_Nhpyrrole","fr_SH","fr_aldehyde","fr_alkyl_carbamate","fr_alkyl_halide","fr_allylic_oxid","fr_amide","fr_amidine","fr_aniline","fr_aryl_methyl","fr_azide","fr_azo","fr_barbitur","fr_benzene","fr_benzodiazepine","fr_bicyclic","fr_diazo","fr_dihydropyridine","fr_epoxide","fr_ester","fr_ether","fr_furan","fr_guanido","fr_halogen","fr_hdrzine","fr_hdrzone","fr_imidazole","fr_imide","fr_isocyan","fr_isothiocyan","fr_ketone","fr_ketone_Topliss","fr_lactam","fr_lactone","fr_methoxy","fr_morpholine","fr_nitrile","fr_nitro","fr_nitro_arom","fr_nitro_arom_nonortho","fr_nitroso","fr_oxazole","fr_oxime","fr_para_hydroxylation","fr_phenol","fr_phenol_noOrthoHbond","fr_phos_acid","fr_phos_ester","fr_piperdine","fr_piperzine","fr_priamide","fr_prisulfonamd","fr_pyridine","fr_quatN","fr_sulfide","fr_sulfonamd","fr_sulfone","fr_term_acetylene","fr_tetrazole","fr_thiazole","fr_thiocyan","fr_thiophene","fr_unbrch_alkane","fr_urea"]
        elif lft == "Graph_based_Signatures":
            columns += ["Acceptor_Count","Aromatic_Count","Donor_Count","Hydrophobe_Count","NegIonizable_Count","PosIonizable_Count","Acceptor:Acceptor-6.00","Acceptor:Aromatic-6.00","Acceptor:Donor-6.00","Acceptor:Hydrophobe-6.00","Acceptor:NegIonizable-6.00","Acceptor:PosIonizable-6.00","Aromatic:Aromatic-6.00","Aromatic:Donor-6.00","Aromatic:Hydrophobe-6.00","Aromatic:NegIonizable-6.00","Aromatic:PosIonizable-6.00","Donor:Donor-6.00","Donor:Hydrophobe-6.00","Donor:NegIonizable-6.00","Donor:PosIonizable-6.00","Hydrophobe:Hydrophobe-6.00","Hydrophobe:NegIonizable-6.00","Hydrophobe:PosIonizable-6.00","NegIonizable:NegIonizable-6.00","NegIonizable:PosIonizable-6.00","PosIonizable:PosIonizable-6.00","Acceptor:Acceptor-4.00","Acceptor:Aromatic-4.00","Acceptor:Donor-4.00","Acceptor:Hydrophobe-4.00","Acceptor:NegIonizable-4.00","Acceptor:PosIonizable-4.00","Aromatic:Aromatic-4.00","Aromatic:Donor-4.00","Aromatic:Hydrophobe-4.00","Aromatic:NegIonizable-4.00","Aromatic:PosIonizable-4.00","Donor:Donor-4.00","Donor:Hydrophobe-4.00","Donor:NegIonizable-4.00","Donor:PosIonizable-4.00","Hydrophobe:Hydrophobe-4.00","Hydrophobe:NegIonizable-4.00","Hydrophobe:PosIonizable-4.00","NegIonizable:NegIonizable-4.00","NegIonizable:PosIonizable-4.00","PosIonizable:PosIonizable-4.00","Acceptor:Acceptor-2.00","Acceptor:Aromatic-2.00","Acceptor:Donor-2.00","Acceptor:Hydrophobe-2.00","Acceptor:NegIonizable-2.00","Acceptor:PosIonizable-2.00","Aromatic:Aromatic-2.00","Aromatic:Donor-2.00","Aromatic:Hydrophobe-2.00","Aromatic:NegIonizable-2.00","Aromatic:PosIonizable-2.00","Donor:Donor-2.00","Donor:Hydrophobe-2.00","Donor:NegIonizable-2.00","Donor:PosIonizable-2.00","Hydrophobe:Hydrophobe-2.00","Hydrophobe:NegIonizable-2.00","Hydrophobe:PosIonizable-2.00","NegIonizable:NegIonizable-2.00","NegIonizable:PosIonizable-2.00","PosIonizable:PosIonizable-2.00"]
        
    mod_dataset_df = None
    try:
        mod_dataset_df = dataset_df[columns]
    except:
        print("Error representation. ")


    return mod_dataset_df

def represent_train_test(list_of_feature_types, training_df, testing_df):    

    columns = []
    for lft in list_of_feature_types:
        if lft == "General_Descriptors":
            columns += ["HeavyAtomCount","MolLogP","NumHeteroatoms","NumRotatableBonds","RingCount","TPSA","LabuteASA","MolWt","FCount","FCount2"]
        elif lft == "Advanced_Descriptors":
            columns += ["BalabanJ","BertzCT","Chi0","Chi0n","Chi0v","Chi1","Chi1n","Chi1v","Chi2n","Chi2v","Chi3n","Chi3v","Chi4n","Chi4v","HallKierAlpha","Kappa1","Kappa2","Kappa3","NHOHCount","NOCount","PEOE_VSA1","PEOE_VSA10","PEOE_VSA11","PEOE_VSA12","PEOE_VSA13","PEOE_VSA14","PEOE_VSA2","PEOE_VSA3","PEOE_VSA4","PEOE_VSA5","PEOE_VSA6","PEOE_VSA7","PEOE_VSA8","PEOE_VSA9","SMR_VSA1","SMR_VSA10","SMR_VSA2","SMR_VSA3","SMR_VSA4","SMR_VSA5","SMR_VSA6","SMR_VSA7","SMR_VSA8","SMR_VSA9","SlogP_VSA1","SlogP_VSA10","SlogP_VSA11","SlogP_VSA12","SlogP_VSA2","SlogP_VSA3","SlogP_VSA4","SlogP_VSA5","SlogP_VSA6","SlogP_VSA7","SlogP_VSA8","SlogP_VSA9","VSA_EState1","VSA_EState10","VSA_EState2","VSA_EState3","VSA_EState4","VSA_EState5","VSA_EState6","VSA_EState7","VSA_EState8","VSA_EState9"]
        elif lft == "Toxicophores":
            columns += ["Tox_1","Tox_2","Tox_3","Tox_4","Tox_5","Tox_6","Tox_7","Tox_8","Tox_9","Tox_10","Tox_11","Tox_12","Tox_13","Tox_14","Tox_15","Tox_16","Tox_17","Tox_18","Tox_19","Tox_20","Tox_21","Tox_22","Tox_23","Tox_24","Tox_25","Tox_26","Tox_27","Tox_28","Tox_29","Tox_30","Tox_31","Tox_32","Tox_33","Tox_34","Tox_35","Tox_36"]
        elif lft == "Fragments":
            columns += ["fr_Al_COO","fr_Al_OH","fr_Al_OH_noTert","fr_ArN","fr_Ar_COO","fr_Ar_N","fr_Ar_NH","fr_Ar_OH","fr_COO","fr_COO2","fr_C_O","fr_C_O_noCOO","fr_C_S","fr_HOCCN","fr_Imine","fr_NH0","fr_NH1","fr_NH2","fr_N_O","fr_Ndealkylation1","fr_Ndealkylation2","fr_Nhpyrrole","fr_SH","fr_aldehyde","fr_alkyl_carbamate","fr_alkyl_halide","fr_allylic_oxid","fr_amide","fr_amidine","fr_aniline","fr_aryl_methyl","fr_azide","fr_azo","fr_barbitur","fr_benzene","fr_benzodiazepine","fr_bicyclic","fr_diazo","fr_dihydropyridine","fr_epoxide","fr_ester","fr_ether","fr_furan","fr_guanido","fr_halogen","fr_hdrzine","fr_hdrzone","fr_imidazole","fr_imide","fr_isocyan","fr_isothiocyan","fr_ketone","fr_ketone_Topliss","fr_lactam","fr_lactone","fr_methoxy","fr_morpholine","fr_nitrile","fr_nitro","fr_nitro_arom","fr_nitro_arom_nonortho","fr_nitroso","fr_oxazole","fr_oxime","fr_para_hydroxylation","fr_phenol","fr_phenol_noOrthoHbond","fr_phos_acid","fr_phos_ester","fr_piperdine","fr_piperzine","fr_priamide","fr_prisulfonamd","fr_pyridine","fr_quatN","fr_sulfide","fr_sulfonamd","fr_sulfone","fr_term_acetylene","fr_tetrazole","fr_thiazole","fr_thiocyan","fr_thiophene","fr_unbrch_alkane","fr_urea"]
        elif lft == "Graph_based_Signatures":
            columns += ["Acceptor_Count","Aromatic_Count","Donor_Count","Hydrophobe_Count","NegIonizable_Count","PosIonizable_Count","Acceptor:Acceptor-6.00","Acceptor:Aromatic-6.00","Acceptor:Donor-6.00","Acceptor:Hydrophobe-6.00","Acceptor:NegIonizable-6.00","Acceptor:PosIonizable-6.00","Aromatic:Aromatic-6.00","Aromatic:Donor-6.00","Aromatic:Hydrophobe-6.00","Aromatic:NegIonizable-6.00","Aromatic:PosIonizable-6.00","Donor:Donor-6.00","Donor:Hydrophobe-6.00","Donor:NegIonizable-6.00","Donor:PosIonizable-6.00","Hydrophobe:Hydrophobe-6.00","Hydrophobe:NegIonizable-6.00","Hydrophobe:PosIonizable-6.00","NegIonizable:NegIonizable-6.00","NegIonizable:PosIonizable-6.00","PosIonizable:PosIonizable-6.00","Acceptor:Acceptor-4.00","Acceptor:Aromatic-4.00","Acceptor:Donor-4.00","Acceptor:Hydrophobe-4.00","Acceptor:NegIonizable-4.00","Acceptor:PosIonizable-4.00","Aromatic:Aromatic-4.00","Aromatic:Donor-4.00","Aromatic:Hydrophobe-4.00","Aromatic:NegIonizable-4.00","Aromatic:PosIonizable-4.00","Donor:Donor-4.00","Donor:Hydrophobe-4.00","Donor:NegIonizable-4.00","Donor:PosIonizable-4.00","Hydrophobe:Hydrophobe-4.00","Hydrophobe:NegIonizable-4.00","Hydrophobe:PosIonizable-4.00","NegIonizable:NegIonizable-4.00","NegIonizable:PosIonizable-4.00","PosIonizable:PosIonizable-4.00","Acceptor:Acceptor-2.00","Acceptor:Aromatic-2.00","Acceptor:Donor-2.00","Acceptor:Hydrophobe-2.00","Acceptor:NegIonizable-2.00","Acceptor:PosIonizable-2.00","Aromatic:Aromatic-2.00","Aromatic:Donor-2.00","Aromatic:Hydrophobe-2.00","Aromatic:NegIonizable-2.00","Aromatic:PosIonizable-2.00","Donor:Donor-2.00","Donor:Hydrophobe-2.00","Donor:NegIonizable-2.00","Donor:PosIonizable-2.00","Hydrophobe:Hydrophobe-2.00","Hydrophobe:NegIonizable-2.00","Hydrophobe:PosIonizable-2.00","NegIonizable:NegIonizable-2.00","NegIonizable:PosIonizable-2.00","PosIonizable:PosIonizable-2.00"]
        
    mod_training_df = None
    mod_testing_df = None
    try:
        mod_training_df = training_df[columns]
        mod_testing_df = testing_df[columns]
    except:
        return training_df, testing_df


    return mod_training_df, mod_testing_df

def normalizer(norm_hp, df):
    try:
        model = Normalizer(norm=norm_hp).fit(df)
        df_np = model.transform(df)

        return pd.DataFrame(df_np, columns = df.columns)
    except:
        return df  

def max_abs_scaler(df):
    try:
        model = MaxAbsScaler().fit(df)
        df_np = model.transform(df)

        return pd.DataFrame(df_np, columns = df.columns)
    except:
        return df  

def min_max_scaler(df):
    try:
        model = MinMaxScaler().fit(df)
        df_np = model.transform(df)

        return pd.DataFrame(df_np, columns = df.columns)
    except:
        return df  

def standard_scaler(with_mean_str, with_std_str, df):
    with_mean_actual = True
    with_std_actual = True

    if with_mean_str == "False":
        with_mean_actual = False
    if with_std_str == "False":
        with_std_actual = False        
    try:
        model = StandardScaler(with_mean=with_mean_actual, with_std=with_std_actual).fit(df)
        df_np = model.transform(df)

        return pd.DataFrame(df_np, columns = df.columns)
    except:
        return df 

def robust_scaler(with_centering_str, with_scaling_str, df):
    with_centering_actual = True
    with_scaling_actual = True

    if with_centering_str == "False":
        with_centering_actual = False
    if with_scaling_str == "False":
        with_scaling_actual = False        
    try:
        model = RobustScaler(with_centering=with_centering_actual, with_scaling=with_scaling_actual).fit(df)
        df_np = model.transform(df)

        return pd.DataFrame(df_np, columns = df.columns)
    except:
        return df


def normalizer_train_test(norm_hp, df1, df2):
    try:
        model = Normalizer(norm=norm_hp).fit(df1)
        df1_np = model.transform(df1)
        df2_np = model.transform(df2)

        return pd.DataFrame(df1_np, columns = df1.columns), pd.DataFrame(df2_np, columns = df2.columns)
    except:
        return df1, df2  

def max_abs_scaler_train_test(df1, df2):
    try:
        model = MaxAbsScaler().fit(df1)
        df1_np = model.transform(df1)
        df2_np = model.transform(df2)

        return pd.DataFrame(df1_np, columns = df1.columns), pd.DataFrame(df2_np, columns = df2.columns)
    except:
        return df1, df2  

def min_max_scaler_train_test(df1, df2):
    try:
        model = MinMaxScaler().fit(df1)
        df1_np = model.transform(df1)
        df2_np = model.transform(df2)

        return pd.DataFrame(df1_np, columns = df1.columns), pd.DataFrame(df2_np, columns = df2.columns)
    except:
        return df1, df2  

def standard_scaler_train_test(with_mean_str, with_std_str, df1, df2):
    with_mean_actual = True
    with_std_actual = True

    if with_mean_str == "False":
        with_mean_actual = False
    if with_std_str == "False":
        with_std_actual = False        
    try:
        model = StandardScaler(with_mean=with_mean_actual, with_std=with_std_actual).fit(df1)
        df1_np = model.transform(df1)
        df2_np = model.transform(df2)

        return pd.DataFrame(df1_np, columns = df1.columns), pd.DataFrame(df2_np, columns = df2.columns)
    except:
        return df1, df2  

def robust_scaler_train_test(with_centering_str, with_scaling_str, df1, df2):
    with_centering_actual = True
    with_scaling_actual = True

    if with_centering_str == "False":
        with_centering_actual = False
    if with_scaling_str == "False":
        with_scaling_actual = False        
    try:
        model = RobustScaler(with_centering=with_centering_actual, with_scaling=with_scaling_actual).fit(df1)
        df1_np = model.transform(df1)
        df2_np = model.transform(df2)

        return pd.DataFrame(df1_np, columns = df1.columns), pd.DataFrame(df2_np, columns = df2.columns)
    except:
        return df1, df2 


def scale(feature_scaling, dataset_df):

    if feature_scaling[0] == "None":
        return dataset_df
    elif feature_scaling[0] == "Normalizer":
        mod_dataset_df = normalizer(str(feature_scaling[1]), dataset_df)
        return mod_dataset_df
    elif feature_scaling[0] == "MinMaxScaler":
        mod_dataset_df = min_max_scaler(dataset_df)
        return mod_dataset_df
    elif feature_scaling[0] == "MaxAbsScaler":
        mod_dataset_df = max_abs_scaler(dataset_df)
        return mod_dataset_df    
    elif feature_scaling[0] == "StandardScaler":
        mod_dataset_df = standard_scaler(feature_scaling[1], feature_scaling[2], dataset_df)
        return mod_dataset_df
    elif feature_scaling[0] == "RobustScaler":
        mod_dataset_df = robust_scaler(feature_scaling[1], feature_scaling[2], dataset_df)
        return mod_dataset_df           
    else:
        return dataset_df

def scale_train_test(feature_scaling, training_dataset_df, testing_dataset_df):

    if feature_scaling[0] == "None":
        return training_dataset_df, testing_dataset_df
    elif feature_scaling[0] == "Normalizer":
        mod_training_dataset_df, mod_testing_dataset_df = normalizer_train_test(str(feature_scaling[1]), training_dataset_df, testing_dataset_df)
        return mod_training_dataset_df, mod_testing_dataset_df
    elif feature_scaling[0] == "MinMaxScaler":
        mod_training_dataset_df, mod_testing_dataset_df = min_max_scaler_train_test(training_dataset_df, testing_dataset_df)
        return mod_training_dataset_df, mod_testing_dataset_df
    elif feature_scaling[0] == "MaxAbsScaler":
        mod_training_dataset_df, mod_testing_dataset_df = max_abs_scaler_train_test(training_dataset_df, testing_dataset_df)
        return mod_training_dataset_df, mod_testing_dataset_df  
    elif feature_scaling[0] == "StandardScaler":
        mod_training_dataset_df, mod_testing_dataset_df = standard_scaler_train_test(feature_scaling[1], feature_scaling[2], training_dataset_df, testing_dataset_df)
        return mod_training_dataset_df, mod_testing_dataset_df
    elif feature_scaling[0] == "RobustScaler":
        mod_training_dataset_df, mod_testing_dataset_df = robust_scaler_train_test(feature_scaling[1], feature_scaling[2], training_dataset_df, testing_dataset_df)
        return mod_training_dataset_df , mod_testing_dataset_df          
    else:
        return training_dataset_df, testing_dataset_df

def scale_train_test(feature_scaling, training_dataset_df, testing_dataset_df):

    if feature_scaling[0] == "None":
        return training_dataset_df, testing_dataset_df
    elif feature_scaling[0] == "Normalizer":
        mod_training_dataset_df, mod_testing_dataset_df = normalizer_train_test(str(feature_scaling[1]), training_dataset_df, testing_dataset_df)
        return mod_training_dataset_df, mod_testing_dataset_df
    elif feature_scaling[0] == "MinMaxScaler":
        mod_training_dataset_df, mod_testing_dataset_df = min_max_scaler_train_test(training_dataset_df, testing_dataset_df)
        return mod_training_dataset_df, mod_testing_dataset_df
    elif feature_scaling[0] == "MaxAbsScaler":
        mod_training_dataset_df, mod_testing_dataset_df = max_abs_scaler_train_test(training_dataset_df, testing_dataset_df)
        return mod_training_dataset_df, mod_testing_dataset_df  
    elif feature_scaling[0] == "StandardScaler":
        mod_training_dataset_df, mod_testing_dataset_df = standard_scaler_train_test(feature_scaling[1], feature_scaling[2], training_dataset_df, testing_dataset_df)
        return mod_training_dataset_df, mod_testing_dataset_df
    elif feature_scaling[0] == "RobustScaler":
        mod_training_dataset_df, mod_testing_dataset_df = robust_scaler_train_test(feature_scaling[1], feature_scaling[2], training_dataset_df, testing_dataset_df)
        return mod_training_dataset_df , mod_testing_dataset_df          
    else:
        return training_dataset_df, testing_dataset_df

def select_fwe(alpha_str, score_function_str, df, label_col):

    score_function_actual = f_classif

    if(score_function_str == "chi2"):
        score_function_actual = chi2       
    
    try:
        model = SelectFwe(score_func=score_function_actual, alpha = float(alpha_str)).fit(df, label_col)
        df_np = model.transform(df)

        cols_idxs = model.get_support(indices=True)
        features_df_new = df.iloc[:,cols_idxs]
    
        return features_df_new
    except Exception as e:
        return df    

def select_fdr(alpha_str, score_function_str, df, label_col):

    score_function_actual = f_classif

    if(score_function_str == "chi2"):
        score_function_actual = chi2       
    
    try:
        model = SelectFdr(score_func=score_function_actual, alpha = float(alpha_str)).fit(df, label_col)
        df_np = model.transform(df)

        cols_idxs = model.get_support(indices=True)
        features_df_new = df.iloc[:,cols_idxs]
    
        return features_df_new
    except Exception as e:
        return df    

def select_fpr(alpha_str, score_function_str, df, label_col):

    score_function_actual = f_classif

    if(score_function_str == "chi2"):
        score_function_actual = chi2       
    
    try:
        model = SelectFpr(score_func=score_function_actual, alpha = float(alpha_str)).fit(df, label_col)
        df_np = model.transform(df)
    
        cols_idxs = model.get_support(indices=True)
        features_df_new = df.iloc[:,cols_idxs]
    
        return features_df_new
    except Exception as e:
        return df    

def select_percentile(percentile_str, score_function_str, df, label_col):
    score_function_actual = f_classif

    if(score_function_str == "chi2"):
        score_function_actual = chi2       
    
    try:
        model = SelectPercentile(score_func=score_function_actual, percentile = int(percentile_str)).fit(df, label_col)
        df_np = model.transform(df)
    
        cols_idxs = model.get_support(indices=True)
        features_df_new = df.iloc[:,cols_idxs]
    
        return features_df_new
    except Exception as e:
        return df

def variance_threshold(thrsh, df, label_col):
    try:
        model =VarianceThreshold(threshold=thrsh).fit(df, label_col)
        df_np = model.transform(df)
    
        cols_idxs = model.get_support(indices=True)
        features_df_new = df.iloc[:,cols_idxs]
    
        return features_df_new
    except Exception as e:
        return df


def select_fwe_train_test(alpha_str, score_function_str, df1, label_col1, df2):

    score_function_actual = f_classif

    if(score_function_str == "chi2"):
        score_function_actual = chi2       
    
    try:
        model = SelectFwe(score_func=score_function_actual, alpha = float(alpha_str)).fit(df1, label_col1)

        cols_idxs = model.get_support(indices=True)
        features_df1_new = df1.iloc[:,cols_idxs]
        features_df2_new = df2.iloc[:,cols_idxs]
    
        return features_df_new, features_df2_new
    except Exception as e:
        return df1, df2    

def select_fdr_train_test(alpha_str, score_function_str, df1, label_col1, df2):

    score_function_actual = f_classif

    if(score_function_str == "chi2"):
        score_function_actual = chi2       
    
    try:
        model = SelectFdr(score_func=score_function_actual, alpha = float(alpha_str)).fit(df1, label_col1)

        cols_idxs = model.get_support(indices=True)
        features_df1_new = df1.iloc[:,cols_idxs]
        features_df2_new = df2.iloc[:,cols_idxs]
    
        return features_df_new, features_df2_new
    except Exception as e:
        return df1, df2   

def select_fpr_train_test(alpha_str, score_function_str, df1, label_col1, df2):

    score_function_actual = f_classif

    if(score_function_str == "chi2"):
        score_function_actual = chi2       
    
    try:
        model = SelectFpr(score_func=score_function_actual, alpha = float(alpha_str)).fit(df1, label_col1)
    
        cols_idxs = model.get_support(indices=True)
        features_df1_new = df1.iloc[:,cols_idxs]
        features_df2_new = df2.iloc[:,cols_idxs]
    
        return features_df_new, features_df2_new
    except Exception as e:
        return df1, df2   

def select_percentile_train_test(percentile_str, score_function_str, df1, label_col1, df2):
    score_function_actual = f_classif

    if(score_function_str == "chi2"):
        score_function_actual = chi2       
    
    try:
        model = SelectPercentile(score_func=score_function_actual, percentile = int(percentile_str)).fit(df1, label_col1)
    
        cols_idxs = model.get_support(indices=True)
        features_df1_new = df1.iloc[:,cols_idxs]
        features_df2_new = df2.iloc[:,cols_idxs]
    
        return features_df_new, features_df2_new
    except Exception as e:
        return df1, df2   

def variance_threshold_train_test(thrsh, df1, label_col1, df2):
    try:
        model =VarianceThreshold(threshold=thrsh).fit(df1, label_col1)
    
        cols_idxs = model.get_support(indices=True)
        features_df1_new = df1.iloc[:,cols_idxs]
        features_df2_new = df2.iloc[:,cols_idxs]
    
        return features_df_new, features_df2_new
    except Exception as e:
        return df1, df2   


def select(feature_selection, dataset_df, label_col):

    if feature_selection[0] == "None":
        return dataset_df
    elif feature_selection[0] == "VarianceThreshold":
        mod_dataset_df = variance_threshold(float(feature_selection[1]), dataset_df, label_col)
        return mod_dataset_df
    elif feature_selection[0] == "SelectPercentile":        
        mod_dataset_df = select_percentile(feature_selection[1], feature_selection[2], dataset_df, label_col)    
        return mod_dataset_df     
    elif feature_selection[0] == "SelectFpr":        
        mod_dataset_df = select_fpr(feature_selection[1], feature_selection[2], dataset_df, label_col)    
        return mod_dataset_df
    elif feature_selection[0] == "SelectFdr":        
        mod_dataset_df = select_fdr(feature_selection[1], feature_selection[2], dataset_df, label_col)    
        return mod_dataset_df
    elif feature_selection[0] == "SelectFwe":        
        mod_dataset_df = select_fwe(feature_selection[1], feature_selection[2], dataset_df, label_col)    
        return mod_dataset_df          
    else:
        return dataset_df

def select_train_test(feature_selection, dataset_df1, label_col1, dataset_df2):

    if feature_selection[0] == "None":
        return dataset_df1, dataset_df2
    elif feature_selection[0] == "VarianceThreshold":
        mod_dataset_df1, mod_dataset_df2  = variance_threshold_train_test(float(feature_selection[1]), dataset_df1, label_col1, dataset_df2)
        return mod_dataset_df1, mod_dataset_df2 
    elif feature_selection[0] == "SelectPercentile":        
        mod_dataset_df1, mod_dataset_df2  = select_percentile_train_test(feature_selection[1], feature_selection[2], dataset_df1, label_col1, dataset_df2)    
        return mod_dataset_df1, mod_dataset_df2      
    elif feature_selection[0] == "SelectFpr":        
        mod_dataset_df1, mod_dataset_df2  = select_fpr_train_test(feature_selection[1], feature_selection[2], dataset_df1, label_col1, dataset_df2)    
        return mod_dataset_df1, mod_dataset_df2 
    elif feature_selection[0] == "SelectFdr":        
        mod_dataset_df1, mod_dataset_df2  = select_fdr_train_test(feature_selection[1], feature_selection[2], dataset_df1, label_col1, dataset_df2)    
        return mod_dataset_df1, mod_dataset_df2 
    elif feature_selection[0] == "SelectFwe":        
        mod_dataset_df1, mod_dataset_df2  = select_fwe_train_test(feature_selection[1], feature_selection[2], dataset_df1, label_col1, dataset_df2)    
        return mod_dataset_df1, mod_dataset_df2          
    else:
        return dataset_df1, dataset_df2

def XGBEvaluation(dataset, n_estimators_str, max_depth_str, max_leaves_str, learning_rate_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)


        
    try:
        clf = XGBClassifier(n_estimators=int(n_estimators_str), max_depth=max_depth_actual, random_state=42, 
                            max_leaves=int(max_leaves_str), learning_rate=float(learning_rate_str), n_jobs=1)        


        y =dataset.iloc[:,-1:]
        X = dataset[dataset.columns[:-1]]
        
        scores = cross_val_score(clf, X, y, cv=5, scoring=make_scorer(matthews_corrcoef))
        mean_scores = scores.mean()
        return mean_scores   
        
    except:
        return 0.0  

def GradientBoostingEvalulation(dataset, n_estimators_str, criterion_str, max_depth_str, min_samples_split_str, min_samples_leaf_str,
                            max_features_str, loss_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)

    max_features_actual = None
    if max_features_str != "None":
        max_features_actual = max_features_str  


        
    try:
        clf = GradientBoostingClassifier(n_estimators=int(n_estimators_str), criterion=criterion_str, max_depth=max_depth_actual, random_state=42, 
                                     min_samples_split=int(min_samples_split_str), 
                                     min_samples_leaf=int(min_samples_split_str), max_features=max_features_actual, loss=loss_str)        


        y =dataset.iloc[:,-1:]
        X = dataset[dataset.columns[:-1]]
        
        scores = cross_val_score(clf, X, y, cv=5, scoring=make_scorer(matthews_corrcoef))
        mean_scores = scores.mean()
        return mean_scores   
        
    except:
        return 0.0  

def ExtraTreesEvalulation(dataset, n_estimators_str, criterion_str, max_depth_str, min_samples_split_str, min_samples_leaf_str,
                            max_features_str, class_weight_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)

    max_features_actual = None
    if max_features_str != "None":
        max_features_actual = max_features_str  

    class_weight_actual = None
    if class_weight_str != "None":
        class_weight_actual = class_weight_str   
 
        
    try:
        clf = ExtraTreesClassifier(n_estimators=int(n_estimators_str), criterion=criterion_str, max_depth=max_depth_actual, n_jobs=1, random_state=42, 
                                     class_weight=class_weight_actual,  min_samples_split=int(min_samples_split_str), 
                                     min_samples_leaf=int(min_samples_split_str), max_features=max_features_actual)        


        y =dataset.iloc[:,-1:]
        X = dataset[dataset.columns[:-1]]
        
        scores = cross_val_score(clf, X, y, cv=5, scoring=make_scorer(matthews_corrcoef))
        mean_scores = scores.mean()

        return mean_scores   
        
    except:
        return 0.0  


def RandomForestEvalulation(dataset, n_estimators_str, criterion_str, max_depth_str, min_samples_split_str, min_samples_leaf_str,
                            max_features_str, class_weight_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)

    max_features_actual = None
    if max_features_str != "None":
        max_features_actual = max_features_str  

    class_weight_actual = None
    if class_weight_str != "None":
        class_weight_actual = class_weight_str   
 
        
    try:
        clf = RandomForestClassifier(n_estimators=int(n_estimators_str), criterion=criterion_str, max_depth=max_depth_actual, n_jobs=1, random_state=42, 
                                     class_weight=class_weight_actual,  min_samples_split=int(min_samples_split_str), 
                                     min_samples_leaf=int(min_samples_split_str), max_features=max_features_actual)        


        y =dataset.iloc[:,-1:]
        X = dataset[dataset.columns[:-1]]
        
        scores = cross_val_score(clf, X, y, cv=5, scoring=make_scorer(matthews_corrcoef))
        mean_scores = scores.mean()

        return mean_scores   
        
    except:
        return 0.0  


def ExtraTreeEvalulation(dataset, criterion_str, splitter_str, max_depth_str, min_samples_split_str, min_samples_leaf_str,
                           max_features_str, class_weight_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)

    max_features_actual = None
    if max_features_str != "None":
        max_features_actual = max_features_str  

    class_weight_actual = None
    if class_weight_str != "None":
        class_weight_actual = class_weight_str   
        
    try:
        clf = ExtraTreeClassifier(criterion=criterion_str, splitter='best', max_depth=max_depth_actual, 
                                     min_samples_split=int(min_samples_split_str), min_samples_leaf=int(min_samples_split_str),                                      
                                     max_features=max_features_actual, random_state=0)      

        y =dataset.iloc[:,-1:]
        X = dataset[dataset.columns[:-1]]
        
        scores = cross_val_score(clf, X, y, cv=5, scoring=make_scorer(matthews_corrcoef))
        mean_scores = scores.mean()

        return mean_scores   
        
    except:
        return 0.0    


def DecisionTreeEvalulation(dataset, criterion_str, splitter_str, max_depth_str, min_samples_split_str, min_samples_leaf_str,
                           max_features_str, class_weight_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)

    max_features_actual = None
    if max_features_str != "None":
        max_features_actual = max_features_str  

    class_weight_actual = None
    if class_weight_str != "None":
        class_weight_actual = class_weight_str   
        
    try:
        clf = DecisionTreeClassifier(criterion=criterion_str, splitter=splitter_str, max_depth=max_depth_actual, 
                                     min_samples_split=int(min_samples_split_str), min_samples_leaf=int(min_samples_split_str), 
                                     max_features=max_features_actual, random_state=0,
                                     class_weight=class_weight_actual)      

        y =dataset.iloc[:,-1:]
        X = dataset[dataset.columns[:-1]]
        
        scores = cross_val_score(clf, X, y, cv=5, scoring=make_scorer(matthews_corrcoef))
        mean_scores = scores.mean()

        return mean_scores   
        
    except:
        return 0.0


def AdaBoostEvaluation(dataset, alg, n_est, lr):
    try:
        clf = AdaBoostClassifier(n_estimators=n_est, learning_rate=lr, algorithm=alg, random_state=0)      

        y =dataset.iloc[:,-1:]
        X = dataset[dataset.columns[:-1]]
        
        scores = cross_val_score(clf, X, y, cv=5, scoring=make_scorer(matthews_corrcoef))
        mean_scores = scores.mean()

        return mean_scores   
        
    except:
        return 0.0

def XGBEvaluation_train_test(training_dataset, testing_dataset, n_estimators_str, max_depth_str, max_leaves_str, learning_rate_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)


        
    try:
        clf = XGBClassifier(n_estimators=int(n_estimators_str), max_depth=max_depth_actual, random_state=42, 
                            max_leaves=int(max_leaves_str), learning_rate=float(learning_rate_str), n_jobs=1)        


        y_train =training_dataset.iloc[:,-1:]
        X_train = training_dataset[training_dataset.columns[:-1]]

        y_test =testing_dataset.iloc[:,-1:]
        X_test = testing_dataset[testing_dataset.columns[:-1]]  

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        score = matthews_corrcoef(np.array(y_test), predictions)
        

        return score   
        
    except:
        return 0.0  

def GradientBoostingEvalulation_train_test(training_dataset, testing_dataset, n_estimators_str, criterion_str, max_depth_str, min_samples_split_str, min_samples_leaf_str,
                            max_features_str, loss_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)

    max_features_actual = None
    if max_features_str != "None":
        max_features_actual = max_features_str  


        
    try:
        clf = GradientBoostingClassifier(n_estimators=int(n_estimators_str), criterion=criterion_str, max_depth=max_depth_actual, random_state=42, 
                                     min_samples_split=int(min_samples_split_str), 
                                     min_samples_leaf=int(min_samples_split_str), max_features=max_features_actual, loss=loss_str)        


        y_train =training_dataset.iloc[:,-1:]
        X_train = training_dataset[training_dataset.columns[:-1]]

        y_test =testing_dataset.iloc[:,-1:]
        X_test = testing_dataset[testing_dataset.columns[:-1]]  

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        score = matthews_corrcoef(np.array(y_test), predictions)
        

        return score   
        
    except:
        return 0.0   

def ExtraTreesEvalulation_train_test(training_dataset, testing_dataset, n_estimators_str, criterion_str, max_depth_str, min_samples_split_str, min_samples_leaf_str,
                            max_features_str, class_weight_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)

    max_features_actual = None
    if max_features_str != "None":
        max_features_actual = max_features_str  

    class_weight_actual = None
    if class_weight_str != "None":
        class_weight_actual = class_weight_str   
 
        
    try:
        clf = ExtraTreesClassifier(n_estimators=int(n_estimators_str), criterion=criterion_str, max_depth=max_depth_actual, n_jobs=1, random_state=42, 
                                     class_weight=class_weight_actual,  min_samples_split=int(min_samples_split_str), 
                                     min_samples_leaf=int(min_samples_split_str), max_features=max_features_actual)        


        y_train =training_dataset.iloc[:,-1:]
        X_train = training_dataset[training_dataset.columns[:-1]]

        y_test =testing_dataset.iloc[:,-1:]
        X_test = testing_dataset[testing_dataset.columns[:-1]]  

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        score = matthews_corrcoef(np.array(y_test), predictions)
        

        return score   
        
    except:
        return 0.0  


def RandomForestEvalulation_train_test(training_dataset, testing_dataset, n_estimators_str, criterion_str, max_depth_str, min_samples_split_str, min_samples_leaf_str,
                            max_features_str, class_weight_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)

    max_features_actual = None
    if max_features_str != "None":
        max_features_actual = max_features_str  

    class_weight_actual = None
    if class_weight_str != "None":
        class_weight_actual = class_weight_str   
 
        
    try:
        clf = RandomForestClassifier(n_estimators=int(n_estimators_str), criterion=criterion_str, max_depth=max_depth_actual, n_jobs=1, random_state=42, 
                                     class_weight=class_weight_actual,  min_samples_split=int(min_samples_split_str), 
                                     min_samples_leaf=int(min_samples_split_str), max_features=max_features_actual)        


        y_train =training_dataset.iloc[:,-1:]
        X_train = training_dataset[training_dataset.columns[:-1]]

        y_test =testing_dataset.iloc[:,-1:]
        X_test = testing_dataset[testing_dataset.columns[:-1]]  

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        score = matthews_corrcoef(np.array(y_test), predictions)
        

        return score   
        
    except Exception as e:
        print(e)
        return 0.0  


def ExtraTreeEvalulation_train_test(training_dataset, testing_dataset, criterion_str, splitter_str, max_depth_str, min_samples_split_str, min_samples_leaf_str,
                           max_features_str, class_weight_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)

    max_features_actual = None
    if max_features_str != "None":
        max_features_actual = max_features_str  

    class_weight_actual = None
    if class_weight_str != "None":
        class_weight_actual = class_weight_str   
        
    try:
        clf = ExtraTreeClassifier(criterion=criterion_str, splitter='best', max_depth=max_depth_actual, 
                                     min_samples_split=int(min_samples_split_str), min_samples_leaf=int(min_samples_split_str),                                      
                                     max_features=max_features_actual, random_state=0)      

        y_train =training_dataset.iloc[:,-1:]
        X_train = training_dataset[training_dataset.columns[:-1]]

        y_test =testing_dataset.iloc[:,-1:]
        X_test = testing_dataset[testing_dataset.columns[:-1]]  

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        score = matthews_corrcoef(np.array(y_test), predictions)
        

        return score   
        
    except:
        return 0.0  

def DecisionTreeEvalulation_train_test(training_dataset, testing_dataset, criterion_str, splitter_str, max_depth_str, min_samples_split_str, min_samples_leaf_str,
                           max_features_str, class_weight_str):
    max_depth_actual = None
    if max_depth_str != "None":
        max_depth_actual = int(max_depth_str)

    max_features_actual = None
    if max_features_str != "None":
        max_features_actual = max_features_str  

    class_weight_actual = None
    if class_weight_str != "None":
        class_weight_actual = class_weight_str   
        
    try:
        clf = DecisionTreeClassifier(criterion=criterion_str, splitter=splitter_str, max_depth=max_depth_actual, 
                                     min_samples_split=int(min_samples_split_str), min_samples_leaf=int(min_samples_split_str), 
                                     max_features=max_features_actual, random_state=0,
                                     class_weight=class_weight_actual)      

        y_train =training_dataset.iloc[:,-1:]
        X_train = training_dataset[training_dataset.columns[:-1]]

        y_test =testing_dataset.iloc[:,-1:]
        X_test = testing_dataset[testing_dataset.columns[:-1]]  

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        score = matthews_corrcoef(np.array(y_test), predictions)
        

        return score   
        
    except:
        return 0.0  


def AdaBoostEvaluation_train_test(training_dataset, testing_dataset, alg, n_est, lr):
    try:
        clf = AdaBoostClassifier(n_estimators=n_est, learning_rate=lr, algorithm=alg, random_state=0)      

        y_train =training_dataset.iloc[:,-1:]
        X_train = training_dataset[training_dataset.columns[:-1]]

        y_test =testing_dataset.iloc[:,-1:]
        X_test = testing_dataset[testing_dataset.columns[:-1]]  

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        score = matthews_corrcoef(np.array(y_test), predictions)
        

        return score   
        
    except:
        return 0.0 

def evaluate_pipeline(ml_algorithm, sel_dataset_df):
    if ml_algorithm[0] == "AdaBoostClassifier":
        return AdaBoostEvaluation(sel_dataset_df, str(ml_algorithm[1]), int(ml_algorithm[2]), float(ml_algorithm[3]))
    elif ml_algorithm[0] == "DecisionTreeClassifier":
        return DecisionTreeEvalulation(sel_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])    
    elif ml_algorithm[0] == "ExtraTreeClassifier":
        return ExtraTreeEvalulation(sel_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
    elif ml_algorithm[0] == "RandomForestClassifier":
        return RandomForestEvalulation(sel_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
    elif ml_algorithm[0] == "ExtraTreesClassifier":
        return ExtraTreesEvalulation(sel_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
    elif ml_algorithm[0] == "GradientBoostingClassifier":
        return GradientBoostingEvalulation(sel_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7]) 
    elif ml_algorithm[0] == "XGBClassifier":
        return XGBEvaluation(sel_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4])         
                        
    else:
        return -1.7889854 

def evaluate_pipeline_train_test(ml_algorithm, sel_training_dataset_df, sel_testing_dataset_df):
    if ml_algorithm[0] == "AdaBoostClassifier":
        return AdaBoostEvaluation_train_test(sel_training_dataset_df, sel_testing_dataset_df, str(ml_algorithm[1]), int(ml_algorithm[2]), float(ml_algorithm[3]))
    elif ml_algorithm[0] == "DecisionTreeClassifier":
        return DecisionTreeEvalulation_train_test(sel_training_dataset_df, sel_testing_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])    
    elif ml_algorithm[0] == "ExtraTreeClassifier":
        return ExtraTreeEvalulation_train_test(sel_training_dataset_df, sel_testing_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
    elif ml_algorithm[0] == "RandomForestClassifier":
        return RandomForestEvalulation_train_test(sel_training_dataset_df, sel_testing_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
    elif ml_algorithm[0] == "ExtraTreesClassifier":
        return ExtraTreesEvalulation_train_test(sel_training_dataset_df, sel_testing_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7])         
    elif ml_algorithm[0] == "GradientBoostingClassifier":
        return GradientBoostingEvalulation_train_test(sel_training_dataset_df, sel_testing_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4], ml_algorithm[5], ml_algorithm[6], ml_algorithm[7]) 
    elif ml_algorithm[0] == "XGBClassifier":
        return XGBEvaluation_train_test(sel_training_dataset_df, sel_testing_dataset_df, ml_algorithm[1], ml_algorithm[2], ml_algorithm[3], ml_algorithm[4])         
                        
    else:
        return -1.7889854

def fitness_function(pipeline, dataset_path, resample, generation):
    try:

        list_hp = pipeline.split("#")

        representation = list_hp[0].split("$")
        feature_scaling = list_hp[1].split("$")
        feature_selection = list_hp[2].split("$")
        ml_algorithm  =  list_hp[3].split("$")    
        
        dataset_df = pd.read_csv(dataset_path, header=0, sep=",")
        dataset_df = dataset_df.sample(random_state=generation, frac = 1.0)

        label_col = dataset_df["CLASS"]
        dataset_df = dataset_df.drop("CLASS", axis=1)

        dataset_df = represent(representation, dataset_df)       
        prep_dataset_df = scale(feature_scaling, dataset_df)
        sel_dataset_df = select(feature_selection, prep_dataset_df, label_col)
        sel_dataset_df["CLASS"] = pd.Series(label_col)
        score = evaluate_pipeline(ml_algorithm, sel_dataset_df)
    
        return score
    except:
        return -1.0 

# Example fitness function (replace with your own)
def fitness_function_train_test(pipeline, training_dataset, testing_dataset):
    try:

        list_hp = pipeline.split("#")

        representation = list_hp[0].split("$")
        feature_scaling = list_hp[1].split("$")
        feature_selection = list_hp[2].split("$")
        ml_algorithm  =  list_hp[3].split("$")    
        
        training_df = pd.read_csv(training_dataset, header=0, sep=",")
        training_label_col = training_df["CLASS"]
        training_df = training_df.drop("CLASS", axis=1)

        testing_df = pd.read_csv(testing_dataset, header=0, sep=",")
        testing_label_col = testing_df["CLASS"]
        testing_df = testing_df.drop("CLASS", axis=1)        

        mod_training_df, mod_testing_df = represent_train_test(representation, training_df, testing_df)       
        prep_training_df, prep_testing_df = scale_train_test(feature_scaling, mod_training_df, mod_testing_df)
        sel_training_df, sel_testing_df = select_train_test(feature_selection, prep_training_df, training_label_col, prep_testing_df)
        sel_training_df["CLASS"] = pd.Series(training_label_col)
        sel_testing_df["CLASS"] = pd.Series(testing_label_col)
        score = evaluate_pipeline_train_test(ml_algorithm, sel_training_df, sel_testing_df)
    
        return score
    except Exception as e:
        print(e)
        return -1.0   

def evaluate_fitness(pipeline, dataset_path, time_budget_minutes_alg_eval, resample, generation):
    start_time = time.time()
    
    score = fitness_function(pipeline, dataset_path, resample, generation)
    # This function should evaluate the fitness of the individual within the time budget
    fitness_value = score 
    
    elapsed_time = time.time() - start_time    
    if elapsed_time > (time_budget_minutes_alg_eval * 60):  # Check if elapsed time exceeds time budget
        fitness_value = 0.0  # Set fitness value to zero if time budget exceeded
        
        
    return fitness_value

def evaluate_population_parallel(population, dataset_path, time_budget_minutes_alg_eval, num_cores, resample, generation):
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = []
        for pipeline in population:
            result = pool.apply_async(evaluate_fitness, (pipeline, dataset_path, time_budget_minutes_alg_eval, resample, generation))
            try:
                fitness_value = result.get(timeout=time_budget_minutes_alg_eval * 60)
                results.append((pipeline, fitness_value))
            except multiprocessing.TimeoutError:
                results.append((pipeline, 0.0))  # Set fitness value to zero and elapsed time to the time budget

    fitness_results = []
    pipelines = []
    for result in results:
        individual, fitness_value = result[0], result[1]
        pipelines.append(individual)
        fitness_results.append(fitness_value)

    return pipelines, fitness_results


if __name__ == '__main__':
    timestr = time.strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="AutoML for PK Prediction")
    parser.add_argument("training_dir", help="Choose input CSV(comma-separated values) format file")
    parser.add_argument("testing_dir", help="Choose input CSV(comma-separated values) format file")
    parser.add_argument("output_dir", help="Choose output CSV(comma-separated values) format file")
    parser.add_argument("-pop_size", help="Population size", default=30, type=int)
    parser.add_argument("-xover_rate", help="Crossover rate", default=0.90, type=float)
    parser.add_argument("-mut_rate", help="Mutaion rate", default=0.10, type=float)
    parser.add_argument("-time_budget_min", help="AutoML time budget (in min)", default=60, type=int)
    parser.add_argument("-time_budget_minutes_alg_eval", help="Algorithm/pipeline time budget (in min)", default=5, type=int)   
    parser.add_argument("-seed", help="Choose the pseudo-random seed to serve as input to the models", default=42, type=int)
    parser.add_argument("-num_cores", help="Choose the pseudo-random seed to serve as input to the models", default=1, type=int)

    args = parser.parse_args()

    grammar = {
        "<Start>": [["<feature_definition>", "#", "<feature_scaling>", "#", "<feature_selection>", "#", "<algorithms>"]],
        "<feature_definition>": [["General_Descriptors"], 
                                 ["Advanced_Descriptors"],
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
