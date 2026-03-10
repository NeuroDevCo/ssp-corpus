"""Util functions."""

from pathlib import Path
from time import time

import polars as pl


def benchmark(func):
    """Print the runtime of a decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time()
        value = func(*args, **kwargs)
        end_time = time()
        run_time = end_time - start_time
        print(f"Finished {func.__name__}() in {run_time:.4f} seconds")
        return value

    return wrapper_timer


def add_missing_headers(path: Path = Path("data", "raw")):
    """Add header to file if missing.

    Args:
        path (Path): Path to the file with missing header.
    """
    for f in path.glob("*.csv"):
        header = ["word", "phon"]

        try:
            d = pl.read_csv(f, has_header=False)

            if all(c in header for c in d.slice(0, 1).to_numpy().tolist()[0]):
                d = d.rename(d.head(1).to_dicts().pop())
                d.write_csv(f)
            elif all(
                c in {k: v for k, v in zip(d.columns, header)}
                for c in d.slice(0, 1).to_numpy().tolist()[0]
            ):
                d = d.slice(1, len(d) - 1)
                d = d.rename({k: v for k, v in zip(d.columns, header)})
                d.write_csv(f)
            elif all("column_" in c for c in d.columns):
                d = d.rename({k: v for k, v in zip(d.columns, header)})
                d.write_csv(f)
            else:
                pass

        except pl.exceptions.NoDataError:
            pl.DataFrame(schema={"word": str, "phon": int}).write_csv(f)


def cast_to_str(x: pl.DataFrame, sep: str = "_") -> pl.DataFrame:
    """Cast list columns to strings for saving as CSV.

    Args:
        x (pl.DataFrame): Polars dataframe to cast to str.
        sep (str): Separator to use to join lists into a str.

    Returns:
        pl.DataFrame: Polars dataframe with list columns casted to str.
    """
    return x.with_columns(
        pl.col(pl.List(pl.Int64), pl.List(pl.String))
        .cast(pl.List(pl.String))
        .list.join(sep)
    )


def add_lang_col(x: pl.DataFrame, col: str) -> pl.DataFrame:
    """Add language column to dataset.

    Args:
        x (pl.DataFrame): Polars dataframe to which to add the language column.
        iso639p3code (str): Language code to populate the values of the new column with.

    Returns:
        pl.DataFrame: Polars dataframe with the added language column.
    """
    nm = "iso639p3code"
    return x.with_columns(pl.lit(col).alias(nm)).select(
        x.columns[0], nm, pl.exclude(x.columns[0], nm)
    )
