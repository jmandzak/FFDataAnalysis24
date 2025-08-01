import pandas as pd
from FootballNameMatcher import match_name

YEAR = 23
YEAR_STRING = f"_{YEAR:02d}"


def get_df(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def convert_fp_names(df: pd.DataFrame) -> pd.DataFrame:
    df["PLAYER NAME"] = df["PLAYER NAME"].apply(
        lambda x: match_name(x, force_last_name_match=True)
    )
    return df


def remove_rows_with_no_name(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["PLAYER NAME"].notna()]


def main():
    df = get_df(f"data/master_sheet{YEAR_STRING}.csv")
    df = convert_fp_names(df)
    df = remove_rows_with_no_name(df)
    df.to_csv(f"data/master_sheet{YEAR_STRING}.csv", index=False)


if __name__ == "__main__":
    main()
