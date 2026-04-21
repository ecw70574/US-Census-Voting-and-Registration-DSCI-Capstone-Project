import pandas as pd
import numpy as np
import re
import glob
import os

DATA_DIR = "clean_data/years"
OUTPUT_DIR = "processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_state_level_data():

    import pandas as pd

    data = [
        # ---------------- 2012 ----------------
        {"year": 2012, "election_type": "presidential", "office": "President",
        "candidate_name": "Barack Obama", "party": "Democrat",
        "birthplace_state": "HI",
        "age_at_election": 51, "incumbent": 1, "approval_rating_pct": 52,
        "gdp_growth_pct": 2.3, "inflation_pct": 2.1, "unemployment_pct": 8.1,
        "popular_vote_pct": 51.1, "electoral_votes": 332},

        {"year": 2012, "election_type": "presidential", "office": "President",
        "candidate_name": "Mitt Romney", "party": "Republican",
        "birthplace_state": "MI",
        "age_at_election": 65, "incumbent": 0, "approval_rating_pct": None,
        "gdp_growth_pct": 2.3, "inflation_pct": 2.1, "unemployment_pct": 8.1,
        "popular_vote_pct": 47.2, "electoral_votes": 206},

        {"year": 2012, "election_type": "presidential", "office": "President",
        "candidate_name": "Gary Johnson", "party": "Libertarian",
        "birthplace_state": "ND",
        "age_at_election": 59, "incumbent": 0, "approval_rating_pct": None,
        "gdp_growth_pct": 2.3, "inflation_pct": 2.1, "unemployment_pct": 8.1,
        "popular_vote_pct": 1.0, "electoral_votes": 0},

        # ---------------- 2016 ----------------
        {"year": 2016, "election_type": "presidential", "office": "President",
        "candidate_name": "Hillary Clinton", "party": "Democrat",
        "birthplace_state": "IL",
        "age_at_election": 69, "incumbent": 0, "approval_rating_pct": None,
        "gdp_growth_pct": 1.8, "inflation_pct": 1.3, "unemployment_pct": 4.9,
        "popular_vote_pct": 48.2, "electoral_votes": 227},

        {"year": 2016, "election_type": "presidential", "office": "President",
        "candidate_name": "Donald Trump", "party": "Republican",
        "birthplace_state": "NY",
        "age_at_election": 70, "incumbent": 0, "approval_rating_pct": None,
        "gdp_growth_pct": 1.8, "inflation_pct": 1.3, "unemployment_pct": 4.9,
        "popular_vote_pct": 46.1, "electoral_votes": 304},

        {"year": 2016, "election_type": "presidential", "office": "President",
        "candidate_name": "Gary Johnson", "party": "Libertarian",
        "birthplace_state": "ND",
        "age_at_election": 63, "incumbent": 0, "approval_rating_pct": None,
        "gdp_growth_pct": 1.8, "inflation_pct": 1.3, "unemployment_pct": 4.9,
        "popular_vote_pct": 3.3, "electoral_votes": 0},

        # ---------------- 2020 ----------------
        {"year": 2020, "election_type": "presidential", "office": "President",
        "candidate_name": "Joe Biden", "party": "Democrat",
        "birthplace_state": "PA",
        "age_at_election": 78, "incumbent": 0, "approval_rating_pct": 56,
        "gdp_growth_pct": -2.2, "inflation_pct": 1.2, "unemployment_pct": 8.1,
        "popular_vote_pct": 51.3, "electoral_votes": 306},

        {"year": 2020, "election_type": "presidential", "office": "President",
        "candidate_name": "Donald Trump", "party": "Republican",
        "birthplace_state": "NY",
        "age_at_election": 74, "incumbent": 1, "approval_rating_pct": 43,
        "gdp_growth_pct": -2.2, "inflation_pct": 1.2, "unemployment_pct": 8.1,
        "popular_vote_pct": 46.8, "electoral_votes": 232},

        {"year": 2020, "election_type": "presidential", "office": "President",
        "candidate_name": "Jo Jorgensen", "party": "Libertarian",
        "birthplace_state": "IL",
        "age_at_election": 63, "incumbent": 0, "approval_rating_pct": None,
        "gdp_growth_pct": -2.2, "inflation_pct": 1.2, "unemployment_pct": 8.1,
        "popular_vote_pct": 1.2, "electoral_votes": 0},

        # ---------------- 2024 ----------------
        {"year": 2024, "election_type": "presidential", "office": "President",
        "candidate_name": "Joe Biden", "party": "Democrat",
        "birthplace_state": "PA",
        "age_at_election": 82, "incumbent": 1, "approval_rating_pct": 41,
        "gdp_growth_pct": 2.5, "inflation_pct": 3.4, "unemployment_pct": 3.9,
        "popular_vote_pct": 51.0, "electoral_votes": 303},

        {"year": 2024, "election_type": "presidential", "office": "President",
        "candidate_name": "Donald Trump", "party": "Republican",
        "birthplace_state": "NY",
        "age_at_election": 78, "incumbent": 0, "approval_rating_pct": 45,
        "gdp_growth_pct": 2.5, "inflation_pct": 3.4, "unemployment_pct": 3.9,
        "popular_vote_pct": 47.5, "electoral_votes": 235},

        {"year": 2024, "election_type": "presidential", "office": "President",
        "candidate_name": "Chase Oliver", "party": "Libertarian",
        "birthplace_state": "TN",
        "age_at_election": 39, "incumbent": 0, "approval_rating_pct": None,
        "gdp_growth_pct": 2.5, "inflation_pct": 3.4, "unemployment_pct": 3.9,
        "popular_vote_pct": 1.3, "electoral_votes": 0},
    ]

    df = pd.DataFrame(data)
    df.to_csv("presidential_dataset_full.csv", index=False)


# presidential table 
def load_presidential_tables():

    pres = pd.read_csv("presidential_dataset_full.csv")

    # YEAR-LEVEL MACRO DATA
    macro_df = pres[[
        "year",
        "gdp_growth_pct",
        "inflation_pct",
        "unemployment_pct",
        "approval_rating_pct"
    ]].drop_duplicates()

    # STATE FLAGS (safe join)
    state_flags = pres[["year", "birthplace_state"]].drop_duplicates()
    state_flags["candidate_flag"] = 1
    state_flags = state_flags.rename(columns={"birthplace_state": "states_encoded"})

    return macro_df, state_flags

def build_election_series():

    pres = pd.read_csv("presidential_dataset_full.csv")

    election = pres[[
        "year",
        "popular_vote_pct",
        "electoral_votes"
    ]].drop_duplicates().sort_values("year")

    # rename for clarity
    election = election.rename(columns={
        "year": "election_year",
        "popular_vote_pct": "vote_pct",
        "electoral_votes": "evotes"
    })

    # create lag structure aligned to panel years
    election_lag = election.copy()

    election_lag["year"] = election_lag["election_year"] + 4

    election_lag = election_lag[[
        "year",
        "vote_pct",
        "evotes"
    ]].rename(columns={
        "vote_pct": "lag_popular_vote_pct",
        "evotes": "lag_electoral_votes"
    })

    return election_lag

# cleaning the columns from the source data 
def clean_microdata(file_path, year):

    df = pd.read_csv(file_path)

    df = df.rename(columns={
        "PRTAGE": "age",
        "PESEX": "sex",
        "PEMARITL": "marital_status",
        "PEEDUCA": "education",
        "HEFAMINC": "family_income",
        "PTDTRACE": "race",
        "PEHSPNON": "hispanic_flag",
        "PWSSWGT": "weight",
        "PRS8": "time_at_curr_address",
        "PRNLFSCH": "curr_student",
        "HETENURE": "lease_type",
        "PRDTOCC1": "job_industry_code",
        #"GEDIV": "geo_region",
        "PEDIPGED": "GED_or_HS",
        "GESTFIPS": "states",
        "PES1": "did_vote"
    })

    df["did_vote"] = df["did_vote"].replace({2: 0})
    df["year"] = year

    df["weight"] = df["weight"].astype("float32")

    return df

# previous feature engineering steps 
def feature_engineering(df, year):

    ##### Feature engineering

    # state code mapping 
    state_codes = {
        "AK": '02',
        "AL": "01",
        "AR": "05",
        "AZ": "04",
        "CA": "06",
        "CO": "08",
        "CT": "09",
        "DE": "10",
        "FL": "12",
        "GA": "13",
        "HI": "15",
        "IA": "19",
        "ID": "16",
        "IL": "17",
        "IN": "18",
        "KS": "20",
        "KY": "21",
        "LA": "22",
        "MA": "25",
        "MD": "24",
        "ME": "23",
        "MI": "26",
        "MN": "27",
        "MO": "29",
        "MS": "28",
        "MT": "30",
        "NC": "37",
        "ND": "38",
        "NE": "31",
        "NH": "33",
        "NJ": "34",
        "NM": "35",
        "NV": "32",
        "NY": "36",
        "OH": "39",
        "OK": "40",
        "OR": "41",
        "PA": "42",
        "RI": "44",
        "SC": "45",
        "SD": "46",
        "TN": "47",
        "TX": "48",
        "UT": "49",
        "VA": "51",
        "VT": "50",
        "WA": "53",
        "WI": "55",
        "WV": "54",
        "WY": "56"
    }
    
    data_subset = df.copy()
    # encoding all of the states
    state_fips_rev = {v: k for k, v in state_codes.items()}
    data_subset["state_coded"] = data_subset["states"].astype(str).str.zfill(2)
    data_subset["states_encoded"] = data_subset["state_coded"].map(state_fips_rev)


    # AGE GROUP
    bins = [18, 25, 35, 45, 55, 65, 100]
    labels = ["18-24","25-34","35-44","45-54","55-64","65+"]

    data_subset["age_group"] = pd.cut(data_subset["age"], bins=bins, labels=labels, right=False)
    data_subset = data_subset.drop(columns=["age"])

    # RACE
    race_map = {
        1: "white", 2: "black", 3: "american_indian",
        4: "asian", 5: "pacific_islander"
    }

    data_subset["race_grouped"] = data_subset["race"].map(race_map).fillna("multiracial")
    data_subset = data_subset.drop(columns=["race"])

    # INCOME
    data_subset["income_group"] = np.select(
        [
            data_subset["family_income"].between(1,6),
            data_subset["family_income"].between(7,10),
            data_subset["family_income"].between(11,13),
            data_subset["family_income"].between(14,15),
            data_subset["family_income"] == 16
        ],
        [
            "low","lower_middle","middle","upper_middle","high"
        ],
        default="unknown"
    )
    data_subset = data_subset.drop(columns=["family_income"])

    # EDUCATION
    data_subset["education_group"] = np.select(
        [
            data_subset["education"].between(31,38),
            data_subset["education"] == 39,
            data_subset["education"] == 40,
            data_subset["education"].isin([41,42]),
            data_subset["education"] == 43,
            data_subset["education"] > 43
        ],
        [
            "less_hs","hs_grad","some_college",
            "associates","bachelors","graduate"
        ],
        default="unknown"
    )
    data_subset = data_subset.drop(columns=["education"])

    data_subset = data_subset[[ "states_encoded","age_group","income_group","education_group","year",
        "sex","marital_status","race_grouped","hispanic_flag",
        "curr_student","lease_type","job_industry_code",
        "GED_or_HS","time_at_curr_address",
        "weight","did_vote"
    ]]

    return data_subset

def get_last_presidential_year(year):
    return year - (year % 4)  # works for 2010–2024 structure

def build_panel(df):

    panel_keys = [
        "states_encoded",
        "age_group",
        "income_group",
        "education_group",
        "year"
    ]

    feature_cols = [
        "sex","marital_status","race_grouped","hispanic_flag",
        "curr_student","lease_type","job_industry_code",
        "GED_or_HS","time_at_curr_address"
    ]

    # -----------------------------
    # ONE-HOT ENCODE FEATURES
    # -----------------------------
    dummies = pd.get_dummies(df, columns=feature_cols)

    # -----------------------------
    # FORCE NUMERIC TYPES
    # -----------------------------
    dummies = dummies.astype({
        col: "float64"
        for col in dummies.columns
        if col not in panel_keys
    })

    dummies["weight"] = pd.to_numeric(
        dummies["weight"], errors="coerce"
    ).fillna(0).astype("float64")

    # -----------------------------
    # APPLY WEIGHTS
    # -----------------------------
    feature_cols_only = [
        c for c in dummies.columns
        if c not in panel_keys + ["weight"]
    ]

    dummies[feature_cols_only] = dummies[feature_cols_only].multiply(
        dummies["weight"], axis=0
    )

    # -----------------------------
    # GROUP AND SUM
    # -----------------------------
    panel = dummies.groupby(panel_keys, observed=False).sum()

    # -----------------------------
    # NORMALIZE → CONVERT TO PROPORTIONS
    # -----------------------------
    denom = panel["weight"].replace(0, np.nan)

    panel[feature_cols_only] = panel[feature_cols_only].div(denom, axis=0)
    panel[feature_cols_only] = panel[feature_cols_only].fillna(0)

    # optional cleanup: remove redundant baseline column if present
    if "did_vote_0" in panel.columns:
        panel = panel.drop(columns=["did_vote_0"])

    return panel.reset_index()

def process_year(file, year, macro_df, state_flags, election_lag):

    df = clean_microdata(file, year)
    df = feature_engineering(df, year)
    panel = build_panel(df)

    # STATE FLAGS (safe join)
    panel = panel.merge(
        state_flags,
        left_on=["year", "states_encoded"],
        right_on=["year", "states_encoded"],
        how="left"
    )

    panel["candidate_flag"] = panel["candidate_flag"].fillna(0)

    # MACRO DATA
    panel = panel.merge(macro_df, on="year", how="left")

    # ELECTION LAGS
    panel = panel.merge(election_lag, on="year", how="left")

    return panel


def extract_year(file):
    match = re.search(r"nov(\d+)pub_clean", file)
    return int(match.group(1)) if match else -1


def run_pipeline():

    load_state_level_data()

    macro_df, state_flags = load_presidential_tables()
    election_lag = build_election_series()

    files = glob.glob(os.path.join(DATA_DIR, "nov*pub_clean.csv"))
    files = sorted(files, key=extract_year)

    all_years = []

    for file in files:
        year = extract_year(file)
        print(f"Processing {year}")

        panel = process_year(
            file,
            year,
            macro_df,
            state_flags,
            election_lag
        )

        panel.to_csv(
            os.path.join(OUTPUT_DIR, f"panel_{year}.csv"),
            index=False
        )

        all_years.append(panel)

    return pd.concat(all_years, ignore_index=True)


final_df = run_pipeline()
#final_df = add_election_lags(final_df)

'''
final_df.to_csv(
    os.path.join(OUTPUT_DIR, "FINAL_PANEL_DATASET.csv"),
    index=False
)
'''

print("Pipeline complete.")