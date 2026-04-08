"""
cps_utils.py
------------
Utilities for loading CPS fixed-width .dat files and comparing columns
across multiple years. Each year has its own documentation with potentially
different column names and positions, so layouts are parsed per-file.
"""

import re
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Parse column layout from a CPS technical documentation .txt file
# ---------------------------------------------------------------------------

def parse_cps_layout(txt_path: str) -> list[dict]:
    """
    Parse column definitions from a pdftotext-extracted CPS documentation file.

    Returns a list of dicts: {name, size, start, end}
    Skips FILLER and PADDING entries (unnamed padding fields).

    To generate the .txt from a PDF:
        $ pdftotext -layout cpsnov18.pdf cpsnov18.txt

    Args:
        txt_path: Path to the pdftotext-extracted documentation .txt file.

    Returns:
        List of column dicts with keys: name, size, start, end.
    """
    pattern = re.compile(
        r'^([A-Z][A-Z0-9]{1,9})\s+(\d+)\s+.+?(\d+)\s*-\s*(\d+)\s*$',
        re.MULTILINE
    )
    with open(txt_path) as f:
        text = f.read()

    cols = []
    for name, size, start, end in pattern.findall(text):
        if name in ("FILLER", "PADDING"):
            continue
        cols.append({
            "name":  name,
            "size":  int(size),
            "start": int(start),
            "end":   int(end),
        })
    return cols


def layout_to_colspecs(layout: list[dict]) -> tuple[list[tuple], list[str]]:
    """
    Convert a parsed layout list into (colspecs, colnames) for pd.read_fwf.
    CPS locations are 1-indexed; pd.read_fwf expects 0-indexed half-open intervals.
    """
    colspecs = [(c["start"] - 1, c["end"]) for c in layout]
    colnames  = [c["name"] for c in layout]
    return colspecs, colnames


# ---------------------------------------------------------------------------
# 2. Load a single CPS .dat file
# ---------------------------------------------------------------------------

def load_cps_dat(
    filepath: str,
    layout: list[dict],
    dtype: str = "str",
) -> pd.DataFrame:
    """
    Load a CPS fixed-width .dat file into a DataFrame.

    Args:
        filepath:  Path to the .dat file.
        layout:    Parsed layout from parse_cps_layout() for that file's year.
        dtype:     dtype passed to read_fwf. Default 'str' preserves leading
                   zeros; cast columns later as needed.

    Returns:
        DataFrame with column names from the layout.
    """
    colspecs, colnames = layout_to_colspecs(layout)
    colnames = [name.upper() for name in colnames]
    return pd.read_fwf(
        filepath,
        colspecs=colspecs,
        names=colnames,
        header=None,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# 3. Build a layout map - one layout per .dat file, each from its own doc .txt
# ---------------------------------------------------------------------------

def build_layout_map(pairs: dict[str, str]) -> dict[str, list[dict]]:
    """
    Parse a layout for each .dat file from its corresponding documentation .txt.

    Since CPS column names and positions change year to year, each file needs
    its own layout parsed from that year's technical documentation.

    Args:
        pairs: Dict mapping dat filename (no path) -> path to its doc .txt file.

               Example:
                   {
                       "cpsnov18pub.dat": "original_data/cpsnov18.txt",
                       "cpsnov20pub.dat": "original_data/cpsnov20.txt",
                       "cpsnov22pub.dat": "original_data/cpsnov22.txt",
                   }

    Returns:
        Dict mapping dat filename -> parsed layout list.
    """
    layout_map = {}
    for dat_fname, txt_path in pairs.items():
        layout = parse_cps_layout(txt_path)
        print(f"  {dat_fname}: {len(layout)} columns parsed from {txt_path}")
        layout_map[dat_fname] = layout
    return layout_map


# ---------------------------------------------------------------------------
# 4. Load all .dat files using their per-year layouts
# ---------------------------------------------------------------------------

def load_all_cps_dat(
    folder: str,
    layout_map: dict[str, list[dict]],
    dtype: str = "str",
) -> dict[str, pd.DataFrame]:
    """
    Load multiple CPS .dat files, each using its own year-specific layout.

    Args:
        folder:     Directory containing the .dat files.
        layout_map: Dict mapping dat filename -> layout (from build_layout_map).
                    Only files listed in layout_map will be loaded.
        dtype:      Passed to load_cps_dat.

    Returns:
        Dict mapping filename -> DataFrame.
    """
    dfs = {}
    for fname, layout in layout_map.items():
        fpath = f"{folder}/{fname}"
        print(f"  Loading {fname} ({len(layout)} columns)...")
        dfs[fname] = load_cps_dat(fpath, layout, dtype=dtype)
    return dfs


# ---------------------------------------------------------------------------
# 5. Compare column names across files vs. a main DataFrame
# ---------------------------------------------------------------------------

def compare_columns(
    main_df: pd.DataFrame,
    other_dfs: dict[str, pd.DataFrame],
    verbose: bool = False,
) -> None:
    """
    Print a summary table comparing column names in other DataFrames
    against a main reference DataFrame.

    Args:
        main_df:    Reference DataFrame (your main/target year).
        other_dfs:  Dict mapping label -> DataFrame to compare against main.
        verbose:    If True, also print the actual column name differences.
    """
    main_cols = set(main_df.columns)
    n_main    = len(main_cols)

    rows = []
    for label, df in other_dfs.items():
        other_cols = set(df.columns)
        matching   = main_cols & other_cols
        only_main  = main_cols - other_cols
        only_other = other_cols - main_cols

        rows.append({
            "File":          label,
            "Main cols":     n_main,
            "Other cols":    len(other_cols),
            "Matching":      len(matching),
            "Match %":       f"{100 * len(matching) / n_main:.1f}%",
            "Only in main":  len(only_main),
            "Only in other": len(only_other),
        })

        if verbose:
            print(f"\n--- {label} ---")
            if only_main:
                print(f"  Only in main  ({len(only_main)}): {sorted(only_main)}")
            if only_other:
                print(f"  Only in other ({len(only_other)}): {sorted(only_other)}")

    summary = pd.DataFrame(rows).set_index("File")
    print("\n" + summary.to_string())

# ---------------------------------------------------------------------------
# 6. Compare column names across files vs. a main DataFrame
# ---------------------------------------------------------------------------
def filter_rows_by_negative_one_pct(
        df,
        threshold=0.2,
          missing_value=-1,
            inplace=False):
    """
    Filter rows where a certain percentage of columns have a specific missing value.
    Args:
        df: DataFrame to filter.
        threshold: Fraction of columns with missing_value to mark row as bad.
        missing_value: The value considered as missing (e.g., -1).
        inplace: If True, drop bad rows from df. If False, return a new filtered DataFrame.
    Returns:
            If inplace=True, returns None (modifies df in place). If inplace=False, returns a new DataFrame with bad rows removed.
        """
    # threshold: fraction of cols with missing_value to mark row as bad
    # e.g. 0.5 means ">= 50% of columns are -1"
    bad = (df == missing_value).sum(axis=1) / df.shape[1] >= threshold
    bad = (df < 0).sum(axis=1) / df.shape[1] >= threshold
    n_bad = int(bad.sum())
    n_total = len(df)
    #print(f"{n_bad} rows out of {n_total} have >= {threshold*100:.1f}% of columns == {missing_value}")
    if inplace:
        df.drop(index=df[bad].index, inplace=True)
        #print(f"dropped {n_bad} rows, remaining {len(df)} rows")
        return df
    df_clean = df.loc[~bad].reset_index(drop=True)
    #print(f"returning cleaned df with {len(df_clean)} rows")
    return df_clean



# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    folder = "original_data"
    main_file = "cpsnov18pub.dat"

    # Step 1: declare each .dat file and its corresponding doc .txt
    pairs = {
        "cpsnov18pub.dat": f"{folder}/cpsnov18.txt",
        "cpsnov20pub.dat": f"{folder}/cpsnov20.txt",
        "cpsnov22pub.dat": f"{folder}/cpsnov22.txt",
    }

    # Step 2: parse each year's layout
    print("Parsing layouts...")
    layout_map = build_layout_map(pairs)

    # Step 3: load main file with its layout
    main_df = load_cps_dat(f"{folder}/{main_file}", layout_map[main_file])
    print(f"\nMain file shape: {main_df.shape}")

    # Step 4: load all other files
    print("\nLoading other files...")
    all_dfs = load_all_cps_dat(folder, layout_map)
    other_dfs = {k: v for k, v in all_dfs.items() if k != main_file}

    # Step 5: compare
    compare_columns(main_df, other_dfs, verbose=True)

