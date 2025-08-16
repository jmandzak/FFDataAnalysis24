import os

import matplotlib.pyplot as plt
import pandas as pd

from utilities import get_master_df, split_by_position

STARTERS_ONLY = True
DROP_ROOKIES = True

STARTERS_ONLY_STR = "_starters_only" if STARTERS_ONLY else ""
DROP_ROOKIES_STR = "_drop_rookies" if DROP_ROOKIES else ""
SAME_YEAR_POINTS = True

PPR = True
PPR_STRING = "ppr" if PPR else "standard"
YEAR = 24
YEAR_STRING = f"20{YEAR}"


def drop_next_year_based_stats(df: pd.DataFrame) -> pd.DataFrame:
    all_cols = df.columns.tolist()
    bad_variables = ["RK", "STD", "BOOM", "BUST", "START", "SOS", "DEPTH", "BEST", "WORST", "TIER", "ADP", "POS_AVG."]
    # drop columns that contain any of the bad variables
    for var in bad_variables:
        all_cols = [col for col in all_cols if var not in col]

    return df[all_cols].reset_index(drop=True)


def get_correlation(df: pd.DataFrame) -> pd.DataFrame:
    # drop PLAYER NAME and POS columns
    df = df.drop(columns=["PLAYER NAME", "POS", "TEAM"])
    # calculate the correlation matrix
    correlation_matrix = df.corr()
    # convert the correlation matrix to a DataFrame
    correlation_df = correlation_matrix.reset_index().melt(id_vars="index")
    # only look for correlations with Final_PPG
    if not SAME_YEAR_POINTS:
        correlation_df = correlation_df[correlation_df["index"] == "Final_PPG"]
    else:
        correlation_df = correlation_df[correlation_df["index"] == "AVG_FAN PTS"]
        # drop rows where variable is AVG_FAN PTS or FAN_PTS
        correlation_df = correlation_df[
            (correlation_df["variable"] != "AVG_FAN PTS")
            & (correlation_df["variable"] != "FAN PTS")
        ]

    assert isinstance(correlation_df, pd.DataFrame)

    return correlation_df


def plot_correlation(correlation_df: pd.DataFrame, position: str) -> None:
    plt.figure(figsize=(10, 6))
    # drop nan values
    correlation_df = correlation_df.dropna()
    # sort by value
    correlation_df = correlation_df.sort_values(by="value", ascending=False)
    # drop the final_ppg row
    correlation_df = correlation_df[correlation_df["variable"] != "Final_PPG"]  # type: ignore
    plt.bar(correlation_df["variable"], correlation_df["value"])
    plt.xticks(rotation=90)
    plt.title(
        f"Correlation with Final PPG{STARTERS_ONLY_STR}{DROP_ROOKIES_STR} {PPR_STRING} {YEAR_STRING} - {position}"
    )
    plt.xlabel("Variables")
    plt.ylabel("Correlation Coefficient")
    # plot a horizontal red line at y=0.5 and y=-0.5
    plt.axhline(y=0.5, color="r", linestyle="--", label="0.5 Threshold")
    plt.axhline(y=-0.5, color="r", linestyle="--", label="-0.5 Threshold")
    # plot another line at y=0.75 and y=-0.75
    plt.axhline(y=0.75, color="g", linestyle="--", label="0.75 Threshold")
    plt.axhline(y=-0.75, color="g", linestyle="--", label="-0.75 Threshold")
    plt.tight_layout()
    # set the y-axis limits to -1 and 1
    plt.ylim(-1, 1)
    # save image
    os.makedirs(f"images/{YEAR_STRING}/{PPR_STRING}", exist_ok=True)
    same_year_str = "" if not SAME_YEAR_POINTS else "_same_year_points"
    plt.savefig(
        f"images/{YEAR_STRING}/{PPR_STRING}/correlation{same_year_str}{STARTERS_ONLY_STR}{DROP_ROOKIES_STR}_{position}.png"
    )


def random_corrections(df: pd.DataFrame) -> pd.DataFrame:
    # this is a place for random corrections noticed during analysis

    # change tim boyle to a QB in the pos column
    df.loc[df["PLAYER NAME"] == "TIM BOYLE", "POS"] = "QB"

    # change drew lock to a QB in the pos column
    df.loc[df["PLAYER NAME"] == "DREW LOCK", "POS"] = "QB"
    return df


def keep_top_n_players(df: pd.DataFrame, n: int) -> pd.DataFrame:
    # keep only the top n players based on Final_PPG
    top_players = df.nlargest(n, "Final_PPG")
    return top_players.reset_index(drop=True)


def drop_rookies(df: pd.DataFrame) -> pd.DataFrame:
    # if AVG_FAN PTS is 0, then it's a rookie
    df = df[df["AVG_FAN PTS"] != 0]  # type: ignore
    return df.reset_index(drop=True)


def main() -> None:
    df = get_master_df(ppr=PPR, year=YEAR)
    position_dfs = split_by_position(df)

    for pos, df in position_dfs.items():
        print(f"Position: {pos}")
        # keep only the top 32 players unless it's a WR, then keep top 64
        if STARTERS_ONLY:
            if pos == "WR":
                df = keep_top_n_players(df, 64)
            else:
                df = keep_top_n_players(df, 32)
        if DROP_ROOKIES:
            df = drop_rookies(df)
        if SAME_YEAR_POINTS:
            df = drop_next_year_based_stats(df)
        correlation_df = get_correlation(df)
        plot_correlation(correlation_df, pos)


if __name__ == "__main__":
    main()
