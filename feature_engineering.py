import pandas as pd
import numpy as np
import re

# ==========================================================
# TEXT AND TEMPORAL HELPERS
# ==========================================================
def lexical_diversity(texts):
    tokens = re.findall(r'\b\w+\b', ' '.join(texts).lower())
    if not tokens:
        return 0
    return len(set(tokens)) / len(tokens)

def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def create_features(df):
    """Creates features and returns the dataframe with features for each user"""

    df = df.copy()
    df["body_len"] = df["body"].str.len().fillna(0)
    df["time_diff"] = df.groupby("author")["created_utc"].diff().dt.total_seconds()

    # Aggregate quantitative features
    features = df.groupby("author").agg({
        "id": "count",
        "ups": "mean",
        "downs": "mean",
        "score": "mean",
        "body_len": "mean",
        "time_diff": "median"
    }).reset_index()

    features.columns = [
        "author", "n_comments", "avg_ups", "avg_downs",
        "avg_score", "avg_body_len", "median_time_diff"
    ]

    # Lexical diversity per author
    word_diversity = (
        df.groupby("author")["body"]
        .apply(lexical_diversity)
        .reset_index(name="lexical_diversity")
    )

    # Temporal posting patterns
    df["hour"] = df["created_utc"].dt.hour
    hour_counts = (
        df.groupby(["author", "hour"])
        .size()
        .unstack(fill_value=0)
        .reset_index()  # <-- Important fix
    )

    # Normalize and compute entropy features
    hour_only = hour_counts.drop(columns=["author"])
    hour_fraction = hour_only.div(hour_only.sum(axis=1), axis=0)

    hour_features = pd.DataFrame({
        "author": hour_counts["author"],
        "active_hours": (hour_fraction > 0).sum(axis=1),
        "activity_entropy": hour_fraction.apply(entropy, axis=1)
    })

    # Merge all feature sets
    user_features = (
        features
        .merge(word_diversity, on="author", how="outer")
        .merge(hour_features, on="author", how="outer")
        .fillna(0)
    )

    feature_cols = [c for c in user_features.columns if c != "author"]
    return user_features, feature_cols