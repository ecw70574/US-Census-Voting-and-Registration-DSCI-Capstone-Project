import pandas as pd
import numpy as np
import glob
import os
import re
import glob
import os

DATA_DIR = "clean_data/years"
OUTPUT_DIR = "processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_and_aggregate(file_path, year):

    data = pd.read_csv(file_path)

    # Rename columns
    data = data.rename(columns={
        "PRTAGE": "age", 
        "PESEX": "sex", 
        "PEMARITL": "marital_status",
        "PEEDUCA": "education",
        "HEFAMINC": "family_income",
        "PRCHLD": "number_of_children",
        'PTDTRACE': 'race',
        'PEHSPNON': 'hispanic_flag',
        'PWSSWGT': 'weight',
        'PRS8': 'time_at_curr_address',
        'PRNLFSCH': 'curr_student', 
        'HETENURE': 'lease_type',
        'PRDTOCC1': "job_industry_code",
        "GEDIV": "geo_region",
        "PEDIPGED": "GED_or_HS",
        "PES1": "did_vote"
    })

    # our target
    data.loc[data['did_vote'] == 2, 'did_vote'] = 0

    # subset to desired columns
    data_subset = data[[
        "age","sex","marital_status","education","family_income",
        "race","hispanic_flag","weight","time_at_curr_address",
        "curr_student","lease_type","job_industry_code","geo_region",
        "GED_or_HS","did_vote"
    ]].copy()


    ##### Feature engineering

    # age bins
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ["18-24","25-34","35-44","45-54","55-64","65+"]

    data_subset["age_group"] = pd.cut(
        data_subset["age"],
        bins=bins,
        labels=labels,
        right=False
    )
    data_subset = data_subset.drop(columns=["age"])

    # Hispanic flag
    data_subset['hispanic_flag'] = data_subset['hispanic_flag'].replace({
        1: "Hispanic",
        2: "Non-Hispanic"
    })

    # Race grouping
    race_map = {
        1: "white_only",
        2: "black_only",
        3: "american_indian_only",
        4: "asian_only",
        5: "PI_only"
    }

    data_subset["race_grouped"] = data_subset["race"].map(race_map)
    data_subset["race_grouped"] = data_subset["race_grouped"].fillna("multiracial")
    data_subset = data_subset.drop(columns=["race"])

    # income grouping
    data_subset["family_income_grouped"] = np.select(
        [
            data_subset["family_income"].between(1,6),
            data_subset["family_income"].between(7,10),
            data_subset["family_income"].between(11,13),
            data_subset["family_income"].between(14,15),
            data_subset["family_income"] == 16
        ],
        [
            "low_income",
            "lower_middle_class",
            "middle_class",
            "upper_middle_class",
            "high_income"
        ],
        default="unknown"
    )
    data_subset = data_subset.drop(columns=["family_income"])

    # education grouping
    data_subset["education_grouped"] = np.select(
        [
            data_subset["education"].between(31,38),
            data_subset["education"] == 39,
            data_subset["education"] == 40,
            data_subset["education"].isin([41,42]),
            data_subset["education"] == 43,
            data_subset["education"] > 43
        ],
        [
            "less_than_highschool",
            "high_school_GED_completed",
            "some_college",
            "associates",
            "bachelors",
            "masters_and_above"
        ],
        default="unkown"
    )
    data_subset = data_subset.drop(columns=["education"])

    # final feature set
    data_subset = data_subset[[
        "time_at_curr_address", "curr_student","lease_type","marital_status",
        "job_industry_code","GED_or_HS","sex","race_grouped",
        "geo_region","education_grouped","age_group",
        "family_income_grouped","weight","did_vote"
    ]]

    data_subset["year"] = year


    ##### Weighted aggregation

    group_cols = [
        "time_at_curr_address","curr_student","lease_type","marital_status",
        "job_industry_code","GED_or_HS","sex","race_grouped","did_vote"
    ]

    dummies = pd.get_dummies(data_subset, columns=group_cols)

    for col in dummies.columns:
        if col.startswith(tuple(group_cols)):
            dummies[col] = dummies[col] * dummies["weight"]

    grouped = dummies.groupby(
        ["geo_region","education_grouped","age_group","family_income_grouped","year"]
    ).sum()

    feature_cols = [
        c for c in grouped.columns
        if any(c.startswith(p) for p in group_cols)
    ]

    denom = grouped["weight"].replace(0, np.nan)
    grouped[feature_cols] = grouped[feature_cols].div(denom, axis=0)
    grouped[feature_cols] = grouped[feature_cols].fillna(0)

    if "did_vote_0" in grouped.columns:
        grouped = grouped.drop(columns=["did_vote_0"])

    return grouped.reset_index()



# Run pipeline to run over all years 


files = glob.glob(os.path.join(DATA_DIR, "nov*pub_clean.csv"))

def extract_year(file):
    match = re.search(r"nov(\d+)pub_clean", file)
    return int(match.group(1)) if match else -1

files = sorted(files, key=extract_year)

all_years = []

for file in files:
    year = extract_year(file)

    print(f"Processing year: {year}")

    yearly_df = clean_and_aggregate(file, year)

    yearly_path = os.path.join(OUTPUT_DIR, f"aggregated_{year}.csv")
    yearly_df.to_csv(yearly_path, index=False)

    all_years.append(yearly_df)


# Final training dataset with all years combined 

final_df = pd.concat(all_years, ignore_index=True)

final_path = os.path.join(OUTPUT_DIR, "all_years_aggregated.csv")
final_df.to_csv(final_path, index=False)

print("Done. Saved per-year + final dataset.")