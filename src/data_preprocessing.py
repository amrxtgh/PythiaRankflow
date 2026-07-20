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

    sorted_ratings = ratings.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(sorted_ratings) * train_ratio)

    train = sorted_ratings.iloc[:split_idx]
    test = sorted_ratings.iloc[split_idx:]

    return train, test
