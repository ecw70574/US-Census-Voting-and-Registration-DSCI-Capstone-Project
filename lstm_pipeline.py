import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


# split into training and testing 

# train on years <- 2018, validate on 2020-2022, test on 2024
train_data = model_data[model_data["year"] <= 18] # for training
val_data   = model_data[(model_data["year"] > 18) & (model_data["year"] <= 22)] # for tuning/selection
# after final model has been tuned/optimized, avoid data leakage
test_data  = model_data[model_data["year"] == 24] # for testing performance on unseen observations 