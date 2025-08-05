from sklearn.preprocessing import Normalizer, MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
import fcntl
import pandas as pd

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
