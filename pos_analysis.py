import sys
import typing

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd

from utilities import get_master_df

PPR = False
YEAR = 23


def set_window_position() -> None:
    # Set the position of the matplotlib window to the top left corner of the screen
    manager = plt.get_current_fig_manager()
    if hasattr(manager, "window"):
        manager.window.move(100, 100)  # type: ignore
    else:
        print(
            "Warning: Unable to set window position. This may not work in all environments."
        )


def add_final_finish_to_old_df(
    old_df: pd.DataFrame, final_df: pd.DataFrame
) -> pd.DataFrame:
    # create a dictionary to map player names to their final finish
    # ignore total points and only use PPG. We don't predict injuries
    final_ppg_dict = final_df.set_index("Player")["AVG"].to_dict()
    # add a new column to the old_df with the final finish
    old_df["Final_PPG"] = old_df["PLAYER NAME"].map(final_ppg_dict)
    # drop rows where Final_PPG is NaN
    old_df = old_df[old_df["Final_PPG"].notna()]
    # any value that has %, remove the % sign and convert to float
    old_df = old_df.replace("%", "", regex=True)
    return old_df


def split_by_position(df: pd.DataFrame) -> typing.Dict[str, pd.DataFrame]:
    return {pos: df[df["POS"] == pos] for pos in df["POS"].unique()}


def plot_by_feature(df: pd.DataFrame) -> None:
    # get all columns that are numeric and don't have NaN values
    numeric_df = df.select_dtypes(include=["number"]).dropna(axis=1)
    # drop columns that all have 0 values
    numeric_df = numeric_df.loc[:, (numeric_df != 0).any(axis=0)]
    # plot each numeric column against Final_PPG
    for column in numeric_df.columns:
        if column != "Final_PPG":
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(
                numeric_df[column], numeric_df["Final_PPG"], alpha=0.5
            )
            plt.title(f"{column} vs Final PPG")
            plt.xlabel(column)
            plt.ylabel("Final PPG")
            # add a line of best fit
            try:
                z = np.polyfit(numeric_df[column], numeric_df["Final_PPG"], 1)
                p = np.poly1d(z)
                plt.plot(numeric_df[column], p(numeric_df[column]), color="red")
            except Exception as e:
                print(f"Error fitting line for {column}: {e}")
            plt.grid()
            plt.tight_layout()
            set_window_position()
            # Add interactive hover labels using mplcursors
            cursor = mplcursors.cursor(scatter, hover=True)
            player_names = df["PLAYER NAME"].values

            def label_func(sel):
                idx = sel.index
                sel.annotation.set_text(player_names[idx])

            cursor.connect("add", label_func)
            plt.show()


def remove_players_with_no_stats_last_year(df: pd.DataFrame) -> pd.DataFrame:
    # if AVG_FAN PTS is 0 or NaN, remove the player
    df = df[df["AVG_FAN PTS"].notna() & (df["AVG_FAN PTS"] != 0)]
    return df


def top_n_players_by_ppg(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.nlargest(n, "Final_PPG")


def main() -> None:
    assert (
        len(sys.argv) == 2
    ), "Please provide a position as an argument [QB, RB, WR, TE, K]"
    position = sys.argv[1]

    df = get_master_df(ppr=PPR, year=YEAR)
    df = remove_players_with_no_stats_last_year(df)

    pos_df = split_by_position(df).get(position)
    if pos_df is None:
        print(f"No data found for position: {position}")
        return

    if position == "WR":
        pos_df = top_n_players_by_ppg(pos_df, 64)
    else:
        pos_df = top_n_players_by_ppg(pos_df, 32)

    plot_by_feature(pos_df)


if __name__ == "__main__":
    main()
