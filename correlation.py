import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utilities import get_master_df, split_by_position

YEARS = [25, 24, 23]


def drop_next_year_based_stats(df: pd.DataFrame) -> pd.DataFrame:
    all_cols = df.columns.tolist()
    bad_variables = [
        "RK",
        "STD",
        "BOOM",
        "BUST",
        "START",
        "SOS",
        "DEPTH",
        "BEST",
        "WORST",
        "TIER",
        "ADP",
        "POS_AVG.",
    ]
    # drop columns that contain any of the bad variables
    for var in bad_variables:
        all_cols = [col for col in all_cols if var not in col]

    return df[all_cols].reset_index(drop=True)


def get_correlation(df: pd.DataFrame, same_year: bool) -> pd.DataFrame:
    # drop PLAYER NAME and POS columns
    df = df.drop(columns=["PLAYER NAME", "POS", "TEAM"])
    # calculate the correlation matrix
    correlation_matrix = df.corr()
    # convert the correlation matrix to a DataFrame
    correlation_df = correlation_matrix.reset_index().melt(id_vars="index")
    # only look for correlations with Final_PPG
    if not same_year:
        correlation_df = correlation_df[correlation_df["index"] == "Final_PPG"]
        correlation_df = correlation_df[correlation_df["variable"] != "Final_PPG"]
    else:
        correlation_df = correlation_df[correlation_df["index"] == "AVG_FAN PTS"]
        # drop rows where variable is AVG_FAN PTS or FAN_PTS
        correlation_df = correlation_df[
            (correlation_df["variable"] != "AVG_FAN PTS")
            & (correlation_df["variable"] != "FAN PTS")
            & (correlation_df["variable"] != "Final_PPG")
        ]

    assert isinstance(correlation_df, pd.DataFrame)

    return correlation_df


def plot_correlation(
    correlation_df: pd.DataFrame,
    position: str,
    same_year: bool,
    year: int,
    ppr: bool,
    starters_only: bool,
    should_drop_rookies: bool,
) -> None:
    # drop nan values
    correlation_df = correlation_df.dropna()
    # duplicate value column but make it absolute
    correlation_df = correlation_df.copy()
    correlation_df["abs_value"] = correlation_df["value"].abs()
    # sort by value
    correlation_df = correlation_df.sort_values(by="abs_value", ascending=False)
    plt.figure(figsize=(14, 6))
    bar_heights = np.abs(correlation_df["value"])
    colors = ["green" if val >= 0 else "red" for val in correlation_df["value"]]
    plt.bar(correlation_df["variable"], bar_heights, color=colors)
    plt.xticks(rotation=90)
    plt.xlabel("Variables")
    plt.ylabel("Correlation Coefficient")
    plt.tight_layout()
    # set the y-axis limits to -1 and 1
    plt.ylim(0, 1)
    same_year_str = "Same Year Points" if same_year else "Final PPG"
    plt.title(f"{position} Correlation with {same_year_str}")
    # save image
    ppg_string = "same_year" if same_year else "final_ppg"
    rookies_str = "drop_rookies" if should_drop_rookies else "keep_rookies"
    starters_str = "top_ranked" if starters_only else "all_players"
    player_string = f"{rookies_str}_{starters_str}"
    ppr_string = "ppr" if ppr else "standard"
    os.makedirs(
        f"images/correlation/{year}/{ppr_string}/{ppg_string}/{player_string}",
        exist_ok=True,
    )
    plt.tight_layout()
    plt.savefig(
        f"images/correlation/{year}/{ppr_string}/{ppg_string}/{player_string}/{position}.png"
    )
    plt.close()


def random_corrections(df: pd.DataFrame) -> pd.DataFrame:
    # this is a place for random corrections noticed during analysis

    # change tim boyle to a QB in the pos column
    df.loc[df["PLAYER NAME"] == "TIM BOYLE", "POS"] = "QB"

    # change drew lock to a QB in the pos column
    df.loc[df["PLAYER NAME"] == "DREW LOCK", "POS"] = "QB"
    return df


def keep_top_n_players(df: pd.DataFrame, n: int) -> pd.DataFrame:
    # keep only the top n players based on POS_RK
    top_players = df.nlargest(n, "POS_RK")
    return top_players.reset_index(drop=True)


def drop_rookies(df: pd.DataFrame) -> pd.DataFrame:
    # if AVG_FAN PTS is 0, then it's a rookie
    df = df[df["AVG_FAN PTS"] != 0]  # type: ignore
    return df.reset_index(drop=True)


def process_sample(
    year: int,
    ppr: bool,
    starters_only: bool,
    should_drop_rookies: bool,
    same_year_points: bool,
) -> None:
    df = get_master_df(ppr=ppr, year=year)
    random_corrections(df)
    position_dfs = split_by_position(df)

    for pos, df in position_dfs.items():
        if pos == "UNKNOWN":
            continue
        print(f"Position: {pos}")
        # keep only the top 32 players unless it's a WR, then keep top 64
        if starters_only:
            if pos == "WR":
                df = keep_top_n_players(df, 64)
            else:
                df = keep_top_n_players(df, 32)
        if drop_rookies:
            df = drop_rookies(df)
        if same_year_points:
            df = drop_next_year_based_stats(df)
        correlation_df = get_correlation(df, same_year_points)
        plot_correlation(
            correlation_df,
            pos,
            same_year=same_year_points,
            year=year,
            ppr=ppr,
            starters_only=starters_only,
            should_drop_rookies=should_drop_rookies,
        )


def main() -> None:
    for year in YEARS:
        for ppr in [True, False]:
            for starters_only in [True, False]:
                for should_drop_rookies in [True, False]:
                    for same_year_points in [True, False]:
                        if not same_year_points and year == 25:
                            continue  # don't have final_ppg for year 25 yet
                        process_sample(
                            year,
                            ppr,
                            starters_only,
                            should_drop_rookies,
                            same_year_points,
                        )


if __name__ == "__main__":
    main()
