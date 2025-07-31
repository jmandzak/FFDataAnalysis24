import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from sklearn.decomposition import PCA

from utilities import get_master_df, set_window_position, split_by_position


def is_sos_a_good_deciding_factor(
    df: pd.DataFrame,
    show_comparisons: bool,
    min_sos_difference: int = 10,
    max_rank_difference: int = 5,
) -> None:
    df = df.dropna(subset=["PPR_AVG_RK", "FULL_SOS", "Final_PPG"])
    position = df["POS"].iloc[0]
    if position == "WR":
        # only keep the top 64 WRs based on Final_PPG
        df = df.nlargest(64, "Final_PPG")
    else:
        # only keep the top 32 players for other positions
        df = df.nlargest(32, "Final_PPG")

    # sort df by ppr_avg_rank
    df = df.sort_values(by="PPR_AVG_RK").reset_index(drop=True)

    sos_correct = 0
    sos_wrong = 0

    for i in range(1, len(df)):
        ppr_avg_rank_1 = df.iloc[i - 1]["PPR_AVG_RK"]
        ppr_avg_rank_2 = df.iloc[i]["PPR_AVG_RK"]
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
    print("\n")


def plot_sos(df: pd.DataFrame) -> None:
    set_window_position()
    # drop any row with nans for PPR_AVG_RK or FULL_SOS or Final_PPG
    df = df.dropna(subset=["PPR_AVG_RK", "FULL_SOS", "Final_PPG"])
    position = df["POS"].iloc[0]
    if position == "WR":
        # only keep the top 64 WRs based on Final_PPG
        df = df.nlargest(64, "Final_PPG")
    else:
        # only keep the top 32 players for other positions
        df = df.nlargest(32, "Final_PPG")

    ppr_avg_rank = df["PPR_AVG_RK"]
    full_sos = df["FULL_SOS"]
    final_points = df["Final_PPG"]

    # create a 3d plot with vertical lines for each player
    ax = plt.axes(projection="3d")
    # Normalize colors for the lines
    norm = colors.Normalize(final_points.min(), final_points.max())
    cmap = plt.get_cmap("viridis")
    for x, y, z in zip(ppr_avg_rank, full_sos, final_points):
        color = cmap(norm(z))
        ax.plot([x, x], [y, y], [0, z], color=color, linewidth=2)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(mappable, ax=ax, label="Final PPG")
    ax.set_xlabel("PPR AVG Rank")
    ax.set_ylabel("Full SOS")
    ax.set_zlabel("Final PPG")  # type: ignore

    # Stack and fit PCA
    points = np.column_stack(
        (ppr_avg_rank.to_numpy(), full_sos.to_numpy(), final_points.to_numpy())
    )
    pca = PCA(n_components=1).fit(points)

    center = pca.mean_
    direction = pca.components_[0]

    # Create points along the best-fit line
    t = np.linspace(-1, 1, 100)
    line = center + t[:, np.newaxis] * direction

    # Plot the best-fit line
    ax.plot(line[:, 0], line[:, 1], line[:, 2], color="red", linewidth=2)

    plt.title(f"SOS Analysis - {df['POS'].iloc[0]}")
    plt.show()


def main() -> None:
    master_df = get_master_df()
    position_dfs = split_by_position(master_df)
    for position, df in position_dfs.items():
        print(f"Analyzing position: {position}")
        is_sos_a_good_deciding_factor(df, show_comparisons=False)
        plot_sos(df)


if __name__ == "__main__":
    main()
