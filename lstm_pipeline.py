import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. load data
data = pd.read_csv("lstm_preprocessed_data.csv")

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

surviving_groups = group_survival[group_survival["year_count"] >= 5]["group_id"]

model_data = data[data["group_id"].isin(surviving_groups)].copy()

# 4. sort
model_data = model_data.sort_values(["group_id", "year"])

# 5. fill missing
model_data = model_data.fillna(0)

# 6. features
feature_cols = [
    c for c in model_data.columns
    if c not in ["group_id", "year", "did_vote_1"]
]

# 7. sequences
X_seq, y_seq = [], []

for gid, g in model_data.groupby("group_id"):
    g = g.sort_values("year")

    X_seq.append(g[feature_cols].values)
    y_seq.append(g["did_vote_1"].values)

# 8. padding
X = pad_sequences(X_seq, dtype="float32", padding="post")
y = pad_sequences(y_seq, dtype="float32", padding="post")

# split into training and testing 

# train on years <- 2018, validate on 2020-2022, test on 2024
train_data = model_data[model_data["year"] <= 18]
val_data   = model_data[(model_data["year"] > 18) & (model_data["year"] <= 22)]
test_data  = model_data[model_data["year"] == 24]