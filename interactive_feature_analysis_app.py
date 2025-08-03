import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utilities import get_master_df, split_by_position

POSITIONS = ["QB", "RB", "WR", "TE", "DEF", "K"]


def remove_non_starters(df: pd.DataFrame) -> pd.DataFrame:
    pos = df["POS"].iloc[0]

    if pos == "WR":
        # Only keep the top 64 WRs based on position rank
        return df.nsmallest(64, "POS_RK")
    else:
        # Only keep the top 32 players for other positions
        return df.nsmallest(32, "POS_RK")


def drop_non_relevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    # any rows with NAN in a column, drop those columns
    df = df.dropna(axis=1, how="any")
    # if there are any columns that have 0 for every row, drop those columns
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.drop(columns=["TEAM"], errors="ignore")
    return df


def clean_dfs(position_dfs: dict) -> dict:
    for key, df in position_dfs.items():
        df = remove_non_starters(df)
        df = drop_non_relevant_columns(df)
        position_dfs[key] = df
    return position_dfs


def main() -> None:
    # Load and prepare data
    dfs = {}
    for year in [24, 23]:
        for ppr in [True, False]:
            df = get_master_df(ppr=ppr, year=year)
            position_dfs = split_by_position(df)
            position_dfs = clean_dfs(position_dfs)
            dfs[f"{year}_{'PPR' if ppr else 'Standard'}"] = position_dfs

    st.title("Interactive Feature Analysis")

    # Dropdowns for year and scoring type
    year_options = ["24", "23"]
    scoring_options = ["PPR", "Standard"]

    # Use session_state to persist feature selections
    if "selected_x" not in st.session_state:
        st.session_state.selected_x = None
    if "selected_y" not in st.session_state:
        st.session_state.selected_y = None

    selected_year = st.selectbox("Select Year", year_options, index=0, key="year")
    selected_scoring = st.selectbox(
        "Select Scoring Type", scoring_options, index=0, key="scoring"
    )

    # Map selections to key for dfs dict
    dfs_key = f"{selected_year}_{selected_scoring}"
    position_dfs = dfs[dfs_key]
    positions = POSITIONS

    # Set default axis labels based on scoring type
    default_x = "POS_RK"
    default_y = "Final_PPG"

    # Dropdown for position selection
    selected_position = st.selectbox("Select Position", positions, key="position")
    current_df = position_dfs[selected_position]

    # Get numeric columns for axis selection
    numeric_cols = current_df.select_dtypes(include=["number"]).columns.tolist()

    # Try to keep previous selections if still available
    prev_x = st.session_state.selected_x
    prev_y = st.session_state.selected_y
    x_axis = (
        prev_x
        if prev_x in numeric_cols
        else (default_x if default_x in numeric_cols else numeric_cols[0])
    )
    y_axis = (
        prev_y
        if prev_y in numeric_cols
        else (
            default_y
            if default_y in numeric_cols
            else (numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0])
        )
    )

    # Dropdowns for axis selection
    selected_x = st.selectbox(
        "X Axis Feature", numeric_cols, index=numeric_cols.index(x_axis), key="x_axis"
    )
    selected_y = st.selectbox(
        "Y Axis Feature", numeric_cols, index=numeric_cols.index(y_axis), key="y_axis"
    )

    # Update session_state with current selections
    st.session_state.selected_x = selected_x
    st.session_state.selected_y = selected_y

    # Plotly scatter plot with interactive hover labels
    fig = px.scatter(
        current_df,
        x=selected_x,
        y=selected_y,
        hover_name="PLAYER NAME",
        title=f"{selected_x} vs {selected_y} for {selected_position} ({selected_year} {selected_scoring})",
        opacity=0.7,
    )
    # Add line of best fit
    try:
        # ignore 0 values for fitting
        non_zero_df = current_df[
            (current_df[selected_x] != 0) & (current_df[selected_y] != 0)
        ]
        if non_zero_df.empty:
            st.warning("No data available for line fitting.")
            return
        z = np.polyfit(non_zero_df[selected_x], non_zero_df[selected_y], 1)
        p = np.poly1d(z)
        x_vals = np.linspace(
            non_zero_df[selected_x].min(), non_zero_df[selected_x].max(), 100
        )
        fig.add_traces(
            px.line(
                x=x_vals,
                y=p(x_vals),
                labels={"x": selected_x, "y": selected_y},
            ).data
        )
    except Exception as e:
        st.warning(f"Could not fit line: {e}")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
