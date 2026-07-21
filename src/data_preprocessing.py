# k-core filtering
# Temporal Split

from __future__ import annotations

import pandas as pd

def k_core_filter(
        ratings: pd.DataFrame,
        k: int = 20
        ) -> pd.DataFrame:
    """
    Iteratively remove users and movies/items fewer than k interactions.

    Parameters
    ----------
    ratings: DataFrame
        Ratings DataFrame.

    k: int
        Minimum number of interactions.

    Returns
    -------
    DataFrame
        Filtered dataframe.
    """

    filtered = ratings.copy()

    while True:
        user_counts = filtered["user_id"].value_counts()
        item_counts = filtered["movie_id"].value_counts()

        valid_users = user_counts[user_counts >= k].index
        valid_items = item_counts[item_counts >= k].index

        new_filtered = filtered[
                filtered["user_id"].isin(valid_users) &
                filtered["movie_id"].isin(valid_items)
        ]

        if len(new_filtered) == len(filtered):
            break
        filtered = new_filtered

    return filtered.reset_index(drop=True)

def cold_start_statistics(
        ratings: pd.DataFrame,
        filtered: pd.DataFrame,
        k: int = 20,
    ) -> dict:
    before_users = int((ratings["user_id"].value_counts() < k).sum())
    before_items = int((ratings["movie_id"].value_counts() < k).sum())

    after_users = int((filtered["user_id"].value_counts() < k).sum())
    after_items = int((filtered["movie_id"].value_counts() < k).sum())

    return {
            "before_users": before_users,
            "before_items": before_items,
            "after_users": after_users,
            "after_items": after_items,
            "k": k,
    }
        


def temporal_split(
        ratings: pd.DataFrame,
        train_ratio: float = 0.8
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split ratings by timestamp. Older data trains, newer data tests.

    Parameters
    ----------
    ratings: DataFrame
        Ratings DataFrame with a 'timestamp' column.

    train_ratio: float
        Fraction of data to use for training.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        (train, test) split by time.
    """

    train_frames = []
    test_frames = []

    for _, group in ratings.groupby("user_id"):
        group = group.sort_values("timestamp")
        cutoff = int(len(group) * train_ratio)
        if cutoff == 0:
            continue
        train_frames.append(group.iloc[:cutoff])
        test_frames.append(group.iloc[cutoff:])

    train_df = pd.concat(train_frames).reset_index(drop=True) if train_frames else pd.DataFrame()
    test_df = pd.concat(test_frames).reset_index(drop=True) if test_frames else pd.DataFrame()
    return train_df, test_df

def verify_temporal_split(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        ) -> None:
    """
    Verify no overlap and per-user temporal ordering.
    """
    overlap = train_df.merge(
            test_df,
            on=["user_id", "movie_id", "rating", "timestamp"],
            how="inner",
    )
    assert overlap.empty, f"Train/Test overlap: {len(overlap)} rows"

    for user in train_df["user_id"].unique():
        train_t = train_df.loc[train_df["user_id"] == user, "timestamp"]
        test_t = test_df.loc[test_df["user_id"] == user, "timestamp"]

        if len(test_t) == 0:
            continue

        assert train_t.max() <= test_t.min(), {
                f"Temporal leakage for user {user}: "
                f"train max {train_t.max()} > test min {test_t.min()}"
        }
