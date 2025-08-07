"""Util functions."""

from time import time
from pathlib import Path
import polars as pl


def benchmark(func):
    """Print the runtime of the decorated function"""

    def wrapper_timer(*args, **kwargs):
        start_time = time()
        value = func(*args, **kwargs)
        end_time = time()
        run_time = end_time - start_time
        print(f"Finished {func.__name__}() in {run_time:.4f} seconds")
        return value

    return wrapper_timer


def add_missing_headers(path: Path = Path("data", "raw")):
    """Add header to file if missing."""
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
            pl.DataFrame(schema={"word": str, "phon": int}).write_csv(f.name)
