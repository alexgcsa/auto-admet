from sklearn.feature_selection import VarianceThreshold, SelectPercentile, SelectFpr, SelectFwe, SelectFdr, chi2, f_classif, RFE
from .ml_algorithm_transformer import MLAlgorithmTransformer
import fcntl
import pandas as pd

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
