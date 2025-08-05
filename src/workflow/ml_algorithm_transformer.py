from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


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

