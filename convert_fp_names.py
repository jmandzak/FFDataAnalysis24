import pandas as pd
from FootballNameMatcher import match_name


def get_df(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def convert_fp_names(df: pd.DataFrame) -> pd.DataFrame:
    df["Player"] = df["Player"].apply(
        lambda x: match_name(x, force_last_name_match=True)
    )
    return df


def remove_rows_with_no_name(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Player"].notna()]


def main():
    df = get_df("data/FantasyPros_Fantasy_Football_Points_PPR.csv")
    df = convert_fp_names(df)
    df = remove_rows_with_no_name(df)
    df.to_csv("data/fp_converted_names.csv", index=False)


if __name__ == "__main__":
    main()
