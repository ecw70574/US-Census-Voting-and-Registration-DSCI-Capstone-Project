from cps_utils import parse_cps_layout, build_layout_map, load_cps_dat, load_all_cps_dat, compare_columns, filter_rows_by_negative_one_pct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data2024 = pd.read_csv("clean_data/nov24pub_clean.csv")
data = pd.read_csv('original_data/nov24pub_first_half.csv')
data2 = pd.read_csv('original_data/nov24pub_second_half.csv')
data2024_raw = pd.concat([data, data2], ignore_index=True)

folder_data = "original_data"
folder_text = "txt_files"

pairs = {
    "nov18pub.dat": f"{folder_text}/cpsnov18.txt",
    "nov16pub.dat": f"{folder_text}/cpsnov16.txt",
    "nov14pub.dat": f"{folder_text}/cpsnov14.txt",
    "nov12pub.dat": f"{folder_text}/cpsnov12.txt",
    "nov10pub.dat": f"{folder_text}/cpsnov10.txt"
}

layout_map = build_layout_map(pairs)
all_dfs    = load_all_cps_dat(folder_data, layout_map)

# Load any CSV files in folder_data and add to all_dfs, excluding 2024
EXCLUDE_CSVS = {"nov24pub.csv", "nov24pub_first_half.csv", "nov24pub_second_half.csv"}

for fname in os.listdir(folder_data):
    if fname.endswith(".csv") and fname not in EXCLUDE_CSVS:
        csv_df = pd.read_csv(os.path.join(folder_data, fname))
        all_dfs[fname] = csv_df
        print(f"Loaded CSV: {fname}")

# Capture 2024 PWSSWGT from raw file before the loop
pwsswgt_totals = {}
if 'PWSSWGT' in data2024_raw.columns:
    pwsswgt_totals['2024'] = data2024_raw['PWSSWGT'].sum() / 10000
    print(f"Raw PWSSWGT total for 2024: {pwsswgt_totals['2024']:,.0f}")
else:
    print("WARNING: PWSSWGT not found in nov24pub.csv")

del data2024_raw  # free memory, no longer needed

for fname, df in all_dfs.items():

    print(f"\nStarting {fname}")
    df.columns = df.columns.str.upper()
    df = df.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore') if x.dtype == "object" else x)

    # Capture raw weighted population BEFORE any filtering (4 implied decimal places)
    year = fname.replace('nov', '20').replace('pub.dat', '').replace('pub.csv', '')
    if 'PWSSWGT' in df.columns:
        total_weight = df['PWSSWGT'].sum() / 10000
        pwsswgt_totals[year] = total_weight
        print(f"Raw PWSSWGT total for {year}: {total_weight:,.0f}")
    else:
        print(f"WARNING: PWSSWGT not found in {fname}")

    # Fix PES8 -> PRS8 if mislabeled
    if 'PES8' in df.columns and 'PRS8' not in df.columns:
        df = df.rename(columns={'PES8': 'PRS8'})
        print(f"Renamed PES8 -> PRS8 in {fname}")

    matching_columns = len(set(data2024.columns) & set(df.columns))
    print(f"Number of matching columns: {matching_columns}")
    only_in_data = set(data2024.columns) - set(df.columns)
    print(f"Columns only in data2024: {only_in_data}")
    only_in_data2024 = set(df.columns) - set(data2024.columns)
    print(f"Columns only in df: {only_in_data2024}")
    pct_match_data = matching_columns / len(data2024.columns) * 100
    pct_match_data2024 = matching_columns / len(df.columns) * 100
    print(f"Percent of data columns in data2024: {pct_match_data:.1f}%")
    print(f"Percent of data2024 columns in df: {pct_match_data2024:.1f}%")

    df.loc[df['PES1'] == 1, 'PES2'] = 1
    df.loc[df['PES2'] == 2, 'PES1'] = 2
    df.loc[(df['PES2'] == 1) & (df['PES1'].isna()), 'PES1'] = 2
    print(f"After harmonizing PES1/PES2, shape: {df.shape}")

    print(df['HRINTSTA'].value_counts(dropna=False))
    df = df[df['HRINTSTA'] == 1].reset_index(drop=True)
    print(f"After filtering HRINTSTA==1, shape: {df.shape}")

    df = df[df['PRCITSHP'] != 5].reset_index(drop=True)
    print(f"After filtering PRCITSHP!=5, shape: {df.shape}")

    df = df[df['PRPERTYP'] != 1].reset_index(drop=True)
    print(f"After filtering PRPERTYP!=1, shape: {df.shape}")

    print(df[['PES1', 'PES2']].value_counts(dropna=False))
    df['PES1'] = pd.to_numeric(df['PES1'], errors='coerce').fillna(-1).astype(int)
    df['PES2'] = pd.to_numeric(df['PES2'], errors='coerce').fillna(-1).astype(int)
    df = df[(df['PES1'] >= 0) & (df['PES2'] >= 0)].reset_index(drop=True)
    print(f"After filtering PES1/PES2 >= 0, shape: {df.shape}")

    df = df.drop(columns=[col for col in df.columns if col.startswith('PX') or col.startswith('HX')])
    print(f"After dropping PX/HX columns, shape: {df.shape}")

    cols_to_drop = [col for col in df.columns if (df[col] == -1).all()]
    df = df.drop(columns=cols_to_drop)

    df = df[[col for col in df.columns if col in data2024.columns]]
    print(f"After aligning columns with data2024, shape: {df.shape}")

    # Standardize output filename regardless of input type
    out_name = fname.replace('.dat', '').replace('.csv', '') + '_clean.csv'
    df.to_csv(f"clean_data/{out_name}", index=False)

# Save PWSSWGT totals to a separate file
pwsswgt_df = pd.DataFrame(
    sorted(pwsswgt_totals.items()),
    columns=['year', 'pwsswgt_total']
)
pwsswgt_df.to_csv("clean_data/pwsswgt_totals.csv", index=False)

print("\n=== PWSSWGT Totals by Year ===")
for year, total in sorted(pwsswgt_totals.items()):
    print(f"  {year}: {total:,.0f}")
print("\nSaved PWSSWGT totals to clean_data/pwsswgt_totals.csv")