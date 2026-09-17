"""
data_fetch.py
Pulls free, daily UK gilt yield curve data from the Bank of England's
Interactive Statistical Database (IADB). No API key or signup required.

Series used (all "Daily" frequency, nominal par yields calculated with the
Bank's VRP model, plus the daily Bank Rate as a proxy for the very short end
of the curve):

    IUDBEDR  Official Bank Rate                         -> bank_rate
    IUDSNPY  Nominal par yield, 5 year gilts             -> y5
    IUDMNPY  Nominal par yield, 10 year gilts            -> y10
    IUDLNPY  Nominal par yield, 20 year gilts            -> y20

Together these four points (overnight policy rate, 5y, 10y, 20y) are enough
to describe whether the curve is upward-sloping (normal), roughly flat, or
inverted.
"""

from io import StringIO

import pandas as pd
import requests

BOE_URL = "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"

# Order matters: the BoE CSV export returns columns in the same order the
# series codes were requested in, so we zip them positionally rather than
# trusting exact header text (which can be either the code or a long title
# depending on the export format).
BOE_SERIES = {
    "IUDBEDR": "bank_rate",
    "IUDSNPY": "y5",
    "IUDMNPY": "y10",
    "IUDLNPY": "y20",
}

MATURITY_YEARS = {
    "bank_rate": 0.0,
    "y5": 5.0,
    "y10": 10.0,
    "y20": 20.0,
}


class BoEDataError(RuntimeError):
    """Raised when the Bank of England data can't be fetched or parsed."""


def fetch_boe_yield_data(start_date: str = "01/Jan/2000") -> pd.DataFrame:
    """
    Fetch daily Bank Rate + 5y/10y/20y nominal gilt par yields from the
    Bank of England IADB.

    Returns a DataFrame sorted by date, ascending, with columns:
    date, bank_rate, y5, y10, y20 (all yields in % p.a.)
    """
    params = {
        "csv.x": "yes",
        "Datefrom": start_date,
        "Dateto": "now",
        "SeriesCodes": ",".join(BOE_SERIES.keys()),
        "CSVF": "TT",
        "UsingCodes": "Y",
        "VPD": "Y",
    }

    try:
        resp = requests.get(BOE_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise BoEDataError(
            "Could not reach the Bank of England statistics database. "
            "It may be temporarily down, or your network may be blocking it. "
            f"Underlying error: {exc}"
        ) from exc

    text = resp.text.strip()
    if not text or "<html" in text.lower()[:200]:
        raise BoEDataError(
            "The Bank of England database did not return CSV data (it may have "
            "returned an HTML error page instead). Try again shortly, or check "
            "https://www.bankofengland.co.uk/boeapps/database/ manually."
        )

    try:
        df = pd.read_csv(StringIO(text))
    except Exception as exc:  # noqa: BLE001 - surface any parse failure clearly
        raise BoEDataError(f"Could not parse the response as CSV: {exc}") from exc

    if df.shape[1] < len(BOE_SERIES) + 1:
        raise BoEDataError(
            f"Expected {len(BOE_SERIES) + 1} columns (date + "
            f"{len(BOE_SERIES)} series) but got {df.shape[1]}. The Bank of "
            "England's export format may have changed."
        )

    # First column is always the date; the rest line up positionally with
    # the series codes in the order we requested them.
    df = df.rename(columns={df.columns[0]: "date"})
    value_cols = list(df.columns[1 : 1 + len(BOE_SERIES)])
    df = df.rename(columns=dict(zip(value_cols, BOE_SERIES.values())))
    df = df[["date", *BOE_SERIES.values()]]

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    for col in BOE_SERIES.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.dropna(subset=list(BOE_SERIES.values())).reset_index(drop=True)

    if df.empty:
        raise BoEDataError(
            "Parsed the response but ended up with zero valid rows. The "
            "column layout probably doesn't match what this code expects "
            "any more."
        )

    return df
