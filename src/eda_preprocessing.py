import zipfile
import re
import math
from pathlib import Path
from collections import Counter

import pandas as pd

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Number of rows to sample from the top of the file for in‐memory operations
SAMPLE_NROWS = 3_000_000

# How many rows to read per chunk when streaming
CHUNKSIZE = 50_000

# Substring pattern for our five products (case‐insensitive)
PATTERN = (
    r"credit card|"
    r"personal loan|"
    r"buy now pay later|"
    r"savings account|"
    r"money transfer"
)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_minimal(
    zip_path: Path,
    csv_name: str,
    nrows: int = SAMPLE_NROWS
) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(csv_name) as f:
        return pd.read_csv(
            f,
            usecols=["Product", "Consumer complaint narrative"],
            dtype={"Product": "category"},
            nrows=nrows,
            low_memory=True,
        )


# ─── EDA FUNCTIONS ────────────────────────────────────────────────────────────

def basic_eda(df: pd.DataFrame) -> None:
    print("\n=== DATAFRAME INFO ===")
    print(df.info())
    print("\n=== HEAD ===")
    print(df.head())
    print("\n=== PRODUCT VALUE COUNTS ===")
    print(df["Product"].value_counts(dropna=False))

    df["narr_len"] = (
        df["Consumer complaint narrative"]
        .fillna("")
        .str.split()
        .str.len()
    )
    print("\n=== NARRATIVE LENGTH STATS ===")
    print(df["narr_len"].describe())
    print(f"Empty/missing narratives: {(df['narr_len'] == 0).sum()}")


def basic_eda_streamed(
    zip_path: Path,
    csv_name: str,
    chunksize: int = CHUNKSIZE,
    sample_nrows: int = SAMPLE_NROWS
) -> None:
    prod_counts = Counter()
    total_rows = 0
    missing_narr = 0
    sum_len = 0
    sum_sq = 0
    min_len = math.inf
    max_len = 0

    with zipfile.ZipFile(zip_path, "r") as zf, zf.open(csv_name) as reader:
        for chunk in pd.read_csv(
            reader,
            usecols=["Product", "Consumer complaint narrative"],
            chunksize=chunksize,
            low_memory=True,
        ):
            total_rows += len(chunk)
            prod_counts.update(chunk["Product"].dropna())

            lengths = (
                chunk["Consumer complaint narrative"]
                .fillna("")
                .str.split()
                .str.len()
            )
            missing_narr += (lengths == 0).sum()
            sum_len += lengths.sum()
            sum_sq += (lengths**2).sum()
            min_len = min(min_len, lengths.min())
            max_len = max(max_len, lengths.max())

            if total_rows >= sample_nrows:
                break

    mean_len = sum_len / total_rows
    var_len = (sum_sq / total_rows) - (mean_len**2)
    std_len = math.sqrt(var_len)

    print(f"Total rows processed (sampled): {total_rows}")
    print("Complaints by product:")
    for p, c in prod_counts.most_common():
        print(f"  {p}: {c}")
    print(f"Empty/missing narratives: {missing_narr}")
    print(
        f"Narrative word counts — "
        f"min={min_len}, mean={mean_len:.1f}, std={std_len:.1f}, max={max_len}"
    )


# ─── CLEANING & FILTERING ─────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove boilerplate phrase
    text = re.sub(
        r"i am writing to file a complaint[\s\S]*?consumer financial protection bureau",
        " ",
        text
    )
    # keep letters, numbers, basic punctuation and spaces
    text = re.sub(r"[^a-z0-9\s\.\,\?\!]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["Product"].str.lower().str.contains(PATTERN, na=False)
    df = df[mask].copy()
    df = df[df["Consumer complaint narrative"].notna()]
    df["clean_narrative"] = df["Consumer complaint narrative"].map(clean_text)
    return df[df["clean_narrative"].str.len() > 0]


def chunked_filter_and_clean(
    zip_path: Path,
    csv_name: str,
    output_csv: Path,
    chunksize: int = CHUNKSIZE,
    sample_nrows: int = SAMPLE_NROWS
) -> None:
    first = True
    seen = 0
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf, open(output_csv, "w", newline="") as out_f:
        reader = zf.open(csv_name)
        for chunk in pd.read_csv(
            reader,
            usecols=["Product", "Consumer complaint narrative"],
            chunksize=chunksize,
            low_memory=True,
        ):
            to_take = min(len(chunk), sample_nrows - seen)
            if to_take <= 0:
                break
            sub = chunk.iloc[:to_take]
            seen += to_take

            # filter & clean
            mask = sub["Product"].str.lower().str.contains(PATTERN, na=False)
            sub = sub[mask]
            sub = sub[sub["Consumer complaint narrative"].notna()]
            sub["clean_narrative"] = sub["Consumer complaint narrative"].map(clean_text)
            sub = sub[sub["clean_narrative"].str.len() > 0]

            if not sub.empty:
                sub.to_csv(out_f, index=False, header=first, mode="a")
                first = False
