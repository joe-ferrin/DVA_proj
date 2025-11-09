import duckdb
import pandas as pd
import os

# ==========================================================
# CONNECTION
# ==========================================================
con = duckdb.connect("../data/database.sqlite")

# ==========================================================
# SIMPLE QUERY
# ==========================================================
def simple_query(query, params=None):
    return con.execute(query, params or []).fetchdf()

# ==========================================================
# DATA CLEANING HELPERS
# ==========================================================
def decode(x):
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    return x

def clean_dataframe(df):
    """Decode byte columns, convert timestamps, and remove bad authors."""
    for col in df.columns:
        df[col] = df[col].apply(decode)
    df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
    df = df[df["author"].notnull()]
    df = df[~df["author"].isin(["[deleted]", "AutoModerator"])]
    return df

# ==========================================================
# MAIN DATA RETRIEVAL
# ==========================================================
def load_subreddit_data(subreddits, reload=False):
    """
    Retrieves and cleans data for one or more subreddits.
    Caches parquet files to avoid re-querying.
    Returns combined DataFrame.
    """
    if isinstance(subreddits, str):
        subreddits = [subreddits]

    dfs = []
    for sub in subreddits:
        path = f"../data/{sub}_subreddit.parquet"
        if os.path.exists(path) and not reload:
            print(f"Loading cached data for {sub}...")
            df = pd.read_parquet(path)
        else:
            print(f"Querying data for {sub}...")
            query = "SELECT * FROM May2015 WHERE subreddit = ?"
            df = simple_query(query, [sub])
            df = clean_dataframe(df)
            df.to_parquet(path, engine="pyarrow", index=False, compression="snappy")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {combined.shape[0]} rows from {len(subreddits)} subreddit(s).")
    return combined