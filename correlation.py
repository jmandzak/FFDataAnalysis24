import typing

import matplotlib.pyplot as plt
import pandas as pd

from utilities import get_master_df

STARTERS_ONLY = True
DROP_ROOKIES = True

STARTERS_ONLY_STR = "_starters_only" if STARTERS_ONLY else ""
DROP_ROOKIES_STR = "_drop_rookies" if DROP_ROOKIES else ""

PPR = False
PPR_STRING = "_ppr" if PPR else "_standard"


def split_by_position(df: pd.DataFrame) -> typing.Dict[str, pd.DataFrame]:
    return {pos: df[df["POS"] == pos] for pos in df["POS"].unique()}


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    cols_to_drop = [
        # c
        # for c in cols
        # if "RK" in c
        # or "BEST" in c
        # or "WORST" in c
        # or "TIER" in c
        # or "DEV" in c
        # or "POS_AVG" in c
    ]
    if PPR:
        non_ppr_cols = [
            col for col in cols if not col.startswith("PPR_") and f"PPR_{col}" in cols
        ]
        cols_to_drop.extend(non_ppr_cols)
    else:
        ppr_cols = [col for col in cols if col.startswith("PPR_")]
        cols_to_drop.extend(ppr_cols)

    return df.drop(columns=cols_to_drop)


def get_correlation(df: pd.DataFrame) -> pd.DataFrame:
    # drop PLAYER NAME and POS columns
    df = df.drop(columns=["PLAYER NAME", "POS", "TEAM"])
    # calculate the correlation matrix
    correlation_matrix = df.corr()
    # convert the correlation matrix to a DataFrame
    correlation_df = correlation_matrix.reset_index().melt(id_vars="index")
    # only look for correlations with Final_PPG
    correlation_df = correlation_df[correlation_df["index"] == "Final_PPG"]
    return correlation_df


def plot_correlation(correlation_df: pd.DataFrame, position: str) -> None:
    plt.figure(figsize=(10, 6))
    # drop nan values
    correlation_df = correlation_df.dropna()
    # sort by value
    correlation_df = correlation_df.sort_values(by="value", ascending=False)
    # drop the final_ppg row
    correlation_df = correlation_df[correlation_df["variable"] != "Final_PPG"]
    plt.bar(correlation_df["variable"], correlation_df["value"])
    plt.xticks(rotation=90)
    plt.title(
        f"Correlation with Final PPG{ STARTERS_ONLY_STR}{DROP_ROOKIES_STR}{PPR_STRING} - {position}"
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
    plt.savefig(
        f"images/correlation{STARTERS_ONLY_STR}{DROP_ROOKIES_STR}{PPR_STRING}_{position}.png"
    )
    # plt.show()


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
    df = df[df["AVG_FAN PTS"] != 0]
    return df.reset_index(drop=True)


def main() -> None:
    df = get_master_df(ppr=PPR)
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
        df = drop_irrelevant_columns(df)
        correlation_df = get_correlation(df)
        plot_correlation(correlation_df, pos)


if __name__ == "__main__":
    main()
