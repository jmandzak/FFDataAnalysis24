import os
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
    return old_df


def split_by_position(df: pd.DataFrame) -> typing.Dict[str, pd.DataFrame]:
    return {pos: df[df["POS"] == pos] for pos in df["POS"].unique()}


def _remove_ppr_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Remove PPR specific columns if they exist
    ppr_columns = [col for col in df.columns if "PPR" in col]
    return df.drop(columns=ppr_columns, errors="ignore")


def _set_ppr_columns_for_non_ppr_positions(df: pd.DataFrame) -> pd.DataFrame:
    # for qb, k, and def positions, we need to set the following ppr columns to their equivalent non-ppr columns
    ppr_columns = [
        col
        for col in df.columns
        if "PPR_" in col and col.replace("PPR_", "") in df.columns
    ]
    # split qb, k, and def rows into their own df (make a copy to avoid SettingWithCopyWarning)
    non_ppr_df = df[df["POS"].isin(["QB", "K", "DEF"])]
    non_ppr_df = non_ppr_df.copy()
    # for each ppr column, set the value to the equivalent non-ppr column
    for col in ppr_columns:
        non_ppr_col = col.replace("PPR_", "")
        if non_ppr_col in df.columns:
            non_ppr_df[col] = non_ppr_df[non_ppr_col]

    # remove qb, k, and def rows from the original df
    df = df[~df["POS"].isin(["QB", "K", "DEF"])]
    # concatenate the non-ppr df with the original df
    df = pd.concat([df, non_ppr_df], ignore_index=True)
    return df


def _remove_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Remove Standard specific columns if they exist
    ppr_columns = [col for col in df.columns if "PPR" in col]
    standard_columns = [col.replace("PPR_", "") for col in ppr_columns] + [
        col for col in df.columns if "STANDARD_" in col
    ]
    return df.drop(columns=standard_columns, errors="ignore")


def _remove_ppr_from_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # Remove "PPR_" from column names if it exists
    df.columns = [col.replace("PPR_", "") for col in df.columns]
    return df


def _fix_standard_adp(df: pd.DataFrame) -> pd.DataFrame:
    # change standard_adp to adp
    if "STANDARD_ADP" in df.columns:
        df = df.rename(columns={"STANDARD_ADP": "ADP"})
    return df


def get_master_df(ppr: bool, year: int) -> pd.DataFrame:
    YEAR_STRING = f"_{year}"
    PPR_STRING = "_ppr" if ppr else "_standard"
    current_dir = os.path.dirname(__file__)
    df = pd.read_csv(
        os.path.join(current_dir, "data", f"master_sheet{YEAR_STRING}.csv")
    )
    df = _fix_standard_adp(df)

    final_ppg_file = os.path.join(
        current_dir, "data", f"fp_converted_names{PPR_STRING}{YEAR_STRING}.csv"
    )
    if os.path.exists(final_ppg_file):
        finish_df = pd.read_csv(final_ppg_file)
        master_df = add_final_finish_to_old_df(df, finish_df)
    else:
        master_df = df

    # drop any rows where POS is NaN
    master_df = master_df[master_df["POS"].notna()]
    master_df = master_df.replace("%", "", regex=True)

    if ppr:
        master_df = _set_ppr_columns_for_non_ppr_positions(master_df)
        master_df = _remove_standard_columns(master_df)
        master_df = _remove_ppr_from_column_names(master_df)
    else:
        master_df = _remove_ppr_columns(master_df)

    return master_df
