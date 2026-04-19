import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
# revised methodology: avoids leakage from same group over time

# 1. load data
# reading in resulting CSV from aggregate_and_build.py script 
# data = pd.read_csv("lstm_preprocessed_data.csv")
data = pd.read_csv("all_years_aggregated.csv")

# 2. group id
data["group_id"] = (
    data["geo_region"].astype(str) + "_" +
    data["education_grouped"].astype(str) + "_" +
    data["age_group"].astype(str) + "_" +
    data["family_income_grouped"].astype(str)
)

# 3. survival filter
group_survival = (
    data.groupby("group_id")["year"]
    .nunique()
    .reset_index(name='year_count')
)

# making sure groups are consistent between all 8 years 
# true time series will track same groups over time 
surviving_groups = group_survival[group_survival["year_count"] >= 8]["group_id"]

model_data = data[data["group_id"].isin(surviving_groups)].copy()

# 4. sort
model_data = model_data.sort_values(["group_id", "year"])

# 5. fill missing
model_data = model_data.fillna(0)

# 6. features
# everything except the group id, year, and target variable 
feature_cols = [
    c for c in model_data.columns
    if c not in ["group_id", "year", "did_vote_1"]
]

# 7. buidling sequences for each group
X_seq, y_seq = [], []

for gid, g in model_data.groupby("group_id"):
    g = g.sort_values("year")

    X_seq.append(g[feature_cols].values) # creating a sequence of features per group 
    y_seq.append(g["did_vote_1"].values) # creating a sequence of target feature value per group
# each group now has a sequence of X and y values  

# 8. padding
# add zeros at end if uneven time series lengths between groups
# SHOULDN"T NEED IF WE ALREADY HAVE USED THE GROUP SURVIVAL 
# already ensured same 8 years are non-missing for each group 
# X = pad_sequences(X_seq, dtype="float32", padding="post") 
# y = pad_sequences(y_seq, dtype="float32", padding="post")
X = np.array(X_seq)
y = np.array(y_seq)

# map from group to index 
group_ids = list(model_data["group_id"].unique())
group_to_idx = {g:i for i, g in enumerate(group_ids)}

# split sequences by group identity
# not by rows
train_groups = model_data[model_data["year"] <= 2018]["group_id"].unique()
val_groups   = model_data[(model_data["year"] > 2018) & (model_data["year"] <= 2022)]["group_id"].unique()
test_groups  = model_data[model_data["year"] == 2024]["group_id"].unique()

# group IDs to indices
train_idx = [group_to_idx[g] for g in train_groups if g in group_to_idx]
val_idx   = [group_to_idx[g] for g in val_groups if g in group_to_idx]
test_idx  = [group_to_idx[g] for g in test_groups if g in group_to_idx]

# slice sequence arrays 
# avoid temporal data leakage 
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val     = X[val_idx], y[val_idx]
X_test, y_test   = X[test_idx], y[test_idx]

# split into training and testing 

# train on years <- 2018, validate on 2020-2022, test on 2024
# train_data = model_data[model_data["year"] <= 18] # for training
# val_data   = model_data[(model_data["year"] > 18) & (model_data["year"] <= 22)] # for tuning/selection
# after final model has been tuned/optimized, avoid data leakage
# test_data  = model_data[model_data["year"] == 24] # for testing performance on unseen observations 