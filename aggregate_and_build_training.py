import pandas as pd
import numpy as np
import re
import glob
import os

DATA_DIR = "clean_data/years"
OUTPUT_DIR = "processed/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_two_digit_year(year):
    """
    Converts 4-digit election years to 2-digit format:
    2012 → 12, 2016 → 16, etc.
    """
    return year % 100

def build_election_tables():

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

    # FORCE 2-DIGIT YEAR STANDARD
    df["year"] = df["year"].apply(to_two_digit_year)

    year_level = df.groupby("year").agg({
        "gdp_growth_pct": "first",
        "inflation_pct": "first",
        "unemployment_pct": "first",
        "incumbent": "max"
    }).reset_index()

    candidate_level = df[[
        "year",
        "candidate_name",
        "party",
        "birthplace_state",
        "age_at_election",
        "popular_vote_pct",
        "electoral_votes",
        "incumbent"
    ]].copy()

    return year_level, candidate_level

def build_lags(candidate_level):

    lag = candidate_level.copy()

    lag["year"] = lag["year"] + 4   # still valid in 2-digit system

    return lag.rename(columns={
        "popular_vote_pct": "lag_popular_vote_pct",
        "electoral_votes": "lag_electoral_votes"
    })[["year", "lag_popular_vote_pct", "lag_electoral_votes"]]


def clean_microdata(file_path, year):

    df = pd.read_csv(file_path)

    df = df.rename(columns={
        "PRTAGE": "age",
        "PESEX": "sex",
        "PTDTRACE": "race",
        "GESTFIPS": "states",
        "PWSSWGT": "weight"
    })

    # KEEP YEAR CONSISTENT (2-digit)
    df["year"] = year

    df["weight"] = df["weight"].astype("float64")

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


def build_panel(df):

    panel_keys = ["states", "age_group", "race_grouped", "sex", "year"]

    dummies = pd.get_dummies(df)

    numeric_cols = [c for c in dummies.columns if c not in panel_keys]

    dummies[numeric_cols] = dummies[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    dummies["weight"] = pd.to_numeric(dummies["weight"], errors="coerce").fillna(0)

    dummies[numeric_cols] = dummies[numeric_cols].multiply(
        dummies["weight"], axis=0
    )

    return dummies.groupby(panel_keys, observed=True).sum().reset_index()

def process_year(file, year, year_level, lagged):

    df = clean_microdata(file, year)
    df = feature_engineering(df)

    panel = build_panel(df)

    panel = panel.merge(year_level, on="year", how="left")
    panel = panel.merge(lagged, on="year", how="left")

    return panel

import os
import re

pattern = re.compile(r"nov(\d+)pub_clean")

def extract_year(file):
    name = os.path.basename(file)
    match = pattern.search(name)
    return int(match.group(1)) if match else -1

def run_pipeline():

    year_level, candidate_level = build_election_tables()
    lagged = build_lags(candidate_level)

    files = sorted(glob.glob(os.path.join(DATA_DIR, "nov*pub_clean.csv")))

    all_panels = []

    for file in files:
        year = extract_year(file)
        print("Processing", year)

        panel = process_year(file, year, year_level, lagged)

        '''
        panel.to_csv(
            os.path.join(OUTPUT_DIR, f"panel_{year}.csv"),
            index=False
        )
        '''

        all_panels.append(panel)

    return pd.concat(all_panels, ignore_index=True)

