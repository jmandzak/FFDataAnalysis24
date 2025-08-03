import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
from matplotlib import colors

from utilities import get_master_df, set_window_position, split_by_position

PPR = True
PPR_STRING = "PPR" if PPR else "Standard"
RANK_STRING = "POS_AVG."

YEAR = 24

MAX_RANK_DIFFERENCE = 3
MIN_SOS_DIFFERENCE = 10


def is_sos_a_good_deciding_factor(
    df: pd.DataFrame,
    show_comparisons: bool,
    min_sos_difference: int = MIN_SOS_DIFFERENCE,
    max_rank_difference: int = MAX_RANK_DIFFERENCE,
) -> None:
    df = df.dropna(subset=["POS_AVG.", "FULL_SOS", "Final_PPG"])
    position = df["POS"].iloc[0]
    if position == "WR":
        # only keep the top 64 WRs based on Final_PPG
        df = df.nlargest(64, "Final_PPG")
    else:
        # only keep the top 32 players for other positions
        df = df.nlargest(32, "Final_PPG")

    # sort df by position rank
    if PPR and (position == "RB" or position == "WR" or position == "TE"):
        df = df.sort_values(by=RANK_STRING).reset_index(drop=True)
    else:
        df = df.sort_values(by="POS_AVG.").reset_index(drop=True)

    sos_correct = 0
    sos_wrong = 0

    for i in range(1, len(df)):
        ppr_avg_rank_1 = df.iloc[i - 1][RANK_STRING]
        ppr_avg_rank_2 = df.iloc[i][RANK_STRING]
        if ppr_avg_rank_2 - ppr_avg_rank_1 > max_rank_difference:
            continue

        sos_1 = df.iloc[i - 1]["FULL_SOS"]
        sos_2 = df.iloc[i]["FULL_SOS"]
        final_ppg_1 = df.iloc[i - 1]["Final_PPG"]
        final_ppg_2 = df.iloc[i]["Final_PPG"]

        player_1 = df.iloc[i - 1]["PLAYER NAME"]
        player_2 = df.iloc[i]["PLAYER NAME"]

        if abs(sos_1 - sos_2) < min_sos_difference:
            continue

        if show_comparisons:
            print(
                f"  {player_1}({ppr_avg_rank_1}) vs {player_2}({ppr_avg_rank_2}), {sos_1} vs {sos_2}, {final_ppg_1} vs {final_ppg_2}"
            )

        if (
            sos_1 < sos_2
            and final_ppg_1 > final_ppg_2
            or sos_2 < sos_1
            and final_ppg_2 > final_ppg_1
        ):
            sos_correct += 1
        else:
            sos_wrong += 1

    print(f"Position: {position}, SOS Correct: {sos_correct}, SOS Wrong: {sos_wrong}")
    print()


def plot_sos(df: pd.DataFrame) -> None:
    set_window_position()
    position = df["POS"].iloc[0]

    # drop any row with nans for PPR_AVG_RK or FULL_SOS or Final_PPG
    df = df.dropna(subset=[RANK_STRING, "FULL_SOS", "Final_PPG"])
    if position == "WR":
        # only keep the top 64 WRs based on rank
        df = df.nsmallest(64, RANK_STRING)
    else:
        # only keep the top 32 players for other positions
        df = df.nsmallest(32, RANK_STRING)

    avg_pos_rank = df[RANK_STRING]
    full_sos = df["FULL_SOS"]
    final_points = df["Final_PPG"]

    # create a 2D scatter plot: x=avg_pos_rank, y=full_sos, color by final_points (PPG)
    ax = plt.gca()
    norm = colors.Normalize(final_points.min(), final_points.max())
    cmap = plt.get_cmap("RdYlGn")
    # Scale point size by PPG (final_points)
    min_size = 40
    max_size = 200
    # Normalize final_points to [min_size, max_size]
    if final_points.max() > final_points.min():
        sizes = min_size + (final_points - final_points.min()) / (
            final_points.max() - final_points.min()
        ) * (max_size - min_size)
    else:
        sizes = np.full_like(final_points, (min_size + max_size) / 2)
    scatter = ax.scatter(
        full_sos,
        avg_pos_rank,
        c=final_points,
        cmap=cmap,
        norm=norm,
        s=sizes,
        edgecolor="k",
        alpha=0.8,
    )
    # Add interactive hover labels using mplcursors
    player_names = df["PLAYER NAME"].values
    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        idx = sel.index
        sel.annotation.set_text(player_names[idx])
        sel.annotation.get_bbox_patch().set(fc="yellow", alpha=0.8)

    cbar = plt.colorbar(scatter, ax=ax, label="Final PPG")
    ax.set_ylabel("AVG Pos Rank")
    ax.set_xlabel("Full SOS")
    plt.title(f"SOS Analysis - {df['POS'].iloc[0]}")
    plt.tight_layout()
    plt.grid()
    plt.show()


def main() -> None:
    master_df = get_master_df(ppr=PPR, year=YEAR)
    position_dfs = split_by_position(master_df)
    print(
        f"Comparing pairs of players with a max position rank difference of {MAX_RANK_DIFFERENCE}"
    )
    print(
        f"Only comparing players with a SoS difference of at least {MIN_SOS_DIFFERENCE}"
    )
    print()
    for position, df in position_dfs.items():
        print(f"Analyzing position: {position} {PPR_STRING} 20{YEAR}")
        is_sos_a_good_deciding_factor(df, show_comparisons=False)
        plot_sos(df)


if __name__ == "__main__":
    main()
