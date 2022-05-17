import os
import pandas as pd
from sklearn import model_selection
import numpy as np


"""/* NOTES:

- Data Folder
    - conditon folder
    - control folder
    - scores.csv
    
- scores.csv
    - MADRS: Montgomery Asberg Depression Rating Scale 
*/"""

def activity_features(condition_dict: dict):
    condition_mean = {}
    condition_deviation = {}
    condition_list = condition_dict.keys()
    for cond in condition_list:
        df = condition_dict[cond]
        df = df[df["activity"] != 0]
        condition_mean[cond.replace(".csv", "")] = round(df["activity"].mean(),3)
        condition_deviation[cond.replace(".csv", "")] = round(df["activity"].std(), 3)
        
    condition_features = {"activity_mean": condition_mean, "activity_std": condition_deviation}
    return condition_features

def add_activity_features(scores: pd.DataFrame, condition_dict: dict):
    scores_df = scores.copy()
    condition_features = activity_features(condition_dict)
    #print(condition_features)
    scores_df["activity_mean"] = scores_df["number"]
    scores_df["activity_std"] = scores_df["number"]
    scores_df = scores_df[:23]
    scores_df = scores_df.replace(to_replace=condition_features)
    scores_df.drop(columns=["number"], inplace=True)
    scores_df = scores_df.astype({"activity_mean": float, "activity_std": float})
    return scores_df

def split_edu_age(scores: pd.DataFrame):
    scores_df = scores.copy()
    scores_df["min_age"] = scores_df["age"].apply(lambda x: int(x.split("-")[0]) if isinstance(x, str) else x)
    scores_df["max_age"] = scores_df["age"].apply(lambda x: int(x.split("-")[1]) if isinstance(x, str) else x)
    scores_df["min_edu"] = scores_df["edu"].apply(lambda x: int(x.split("-")[0]) if isinstance(x, str) else x)
    scores_df["max_edu"] = scores_df["edu"].apply(lambda x: int(x.split("-")[1]) if isinstance(x, str) else x)
    
    scores_df.drop(columns=["edu", "age"], inplace=True)
    
    
    return scores_df
    
if __name__ == "__main__":

    # Read all data
    condition = {}
    control = {}
    for folder in os.listdir("../data"):
        if "scores.csv" in folder:
            scores = pd.read_csv(f"../data/{folder}")
        elif "control" in folder:
            for file in os.listdir(f"../data/{folder}"):
                control[file] = pd.read_csv(f"../data/{folder}/{file}")
        elif "condition" in folder:
            for file in os.listdir(f"../data/{folder}"):
                condition[file] = pd.read_csv(f"../data/{folder}/{file}")

    scores.replace(to_replace=" ", value=np.nan, inplace=True)
    
    
    sco = add_activity_features(scores, condition_dict=condition)
    #sco = split_edu_age(sco)
    #Drop Bipolar I: Not enough cases
    sco = sco[sco["afftype"] != 3]
    #Reset the label for positive classes bipolar disease
    sco["afftype"] = sco["afftype"].replace({2.0: int(0), 1.0: int(1), 3.0: int(2)})
    
    x, y = sco.drop(columns=["afftype"]), sco["afftype"]
    x.to_csv("../data/features.csv", index=False)
    y.to_csv("../data/target.csv", index=False)