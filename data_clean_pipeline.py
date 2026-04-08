from cps_utils import parse_cps_layout, build_layout_map, load_cps_dat, load_all_cps_dat, compare_columns, filter_rows_by_negative_one_pct
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data2024 = pd.read_csv("clean_data/nov24pub_clean.csv")

folder_data = "original_data"
folder_text = "txt_files"
main_file = "nov18pub.dat"

# Map each .dat file to its year's documentation .txt
pairs = {
    "nov18pub.dat": f"{folder_text}/cpsnov18.txt",
    "nov16pub.dat": f"{folder_text}/cpsnov16.txt",
    "nov14pub.dat": f"{folder_text}/cpsnov14.txt",
    "nov12pub.dat": f"{folder_text}/cpsnov12.txt",
    "nov10pub.dat": f"{folder_text}/cpsnov10.txt"
}

layout_map = build_layout_map(pairs)
all_dfs    = load_all_cps_dat(folder_data, layout_map)

for fname, df in all_dfs.items():

    print(f"\nStarting {fname}")
    df.columns = df.columns.str.upper()
    df = df.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
    df = df.apply(lambda x: pd.to_numeric(x, errors='ignore') if x.dtype == "object" else x)

    matching_columns = len(set(data2024.columns) & set(df.columns))
    print(f"Number of matching columns: {matching_columns}")
    # Columns in data but not in data2022
    only_in_data = set(data2024.columns) - set(df.columns)
    print(f"Columns only in data2024: {only_in_data}")

    # Columns in data2024 but not in data
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

    thresholds = np.linspace(0.0, 1.0, 101)
    remaining_counts = []

    # for t in thresholds:
    #     df_tmp = filter_rows_by_negative_one_pct(df, threshold=t)
    #     remaining_counts.append(len(df_tmp))

  

    # plt.figure(figsize=(8, 5))
    # plt.plot(thresholds * 100, remaining_counts, marker="o", markersize=3)
    # plt.xlabel("threshold % of columns == -1")
    # plt.ylabel("rows remaining")
    # plt.title("Rows remaining after filtering at each -1 threshold PostDrop Clean")
    # plt.grid(True)
    # #plt.show()

    df.to_csv(f"clean_data/{fname.replace('.dat', '')}_clean.csv", index=False)










