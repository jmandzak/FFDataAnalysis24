import typing

import matplotlib.pyplot as plt
import pandas as pd


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


def get_master_df(ppr: bool, year: int) -> pd.DataFrame:
    # Load the master sheet for 2024
    YEAR_STRING = f"_{year}"
    PPR_STRING = "_ppr" if ppr else "_standard"
    df = pd.read_csv(f"data/master_sheet{YEAR_STRING}.csv")
    finish_df = pd.read_csv(f"data/fp_converted_names{PPR_STRING}{YEAR_STRING}.csv")

    master_df = add_final_finish_to_old_df(df, finish_df)
    # drop any rows where POS is NaN
    master_df = master_df[master_df["POS"].notna()]

    return master_df
