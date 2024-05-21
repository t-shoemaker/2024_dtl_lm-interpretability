#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import bs4
import pandas as pd


def get_blurb(
    tag: bs4.element.Tag,
    keys: tuple = ("author", "title", "d0", "d1", "published", "isbn"),
) -> dict:
    """Get a book blurb and its associated metadata.

    Parameters
    ----------
    tag : bs4.element.Tag
        Element in the XML
    keys : tuple
        Metadata keys to extract
    """
    blurb = {key: tag.find(key) for key in keys}
    if "title" in keys:
        blurb["text"] = blurb["title"].next_sibling
    book = {key: val.text.strip() for key, val in blurb.items() if val}

    return book


def sample(df: pd.DataFrame, label: str, topn: int) -> pd.DataFrame:
    """Sample the blurb DataFrame to the minimum of top-N label counts.

    Parameters
    ----------
    df : pd.DataFrame
        The blurb DataFrame
    label : str
        Label column
    topn : int
        Top-N labels to select
    """
    counts = df.value_counts(label).head(topn)
    sample = counts.min()
    df = df[df[label].isin(counts.index)]
    df = (
        df
        .groupby(label)
        .apply(lambda group: group.sample(sample), include_groups=False)
        .droplevel(1)
        .reset_index()
    )

    return df


def main(args: argparse.Namespace) -> None:
    """Run the script."""
    with args.infile.open("r") as fin:
        data = fin.read()
    soup = bs4.BeautifulSoup(data, features="lxml")

    blurbs = pd.DataFrame([get_blurb(tag) for tag in soup.find_all("book")])
    blurbs["published"] = pd.to_datetime(blurbs["published"])
    blurbs["isbn"] = blurbs["isbn"].astype(int)
    blurbs.drop_duplicates(inplace=True)

    blurbs = sample(blurbs, args.label, args.topn)
    blurbs.to_parquet(args.outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample book blurbs from the Hamburg LTG dataset"
    )
    parser.add_argument("--infile", type=Path, help="Blurb file")
    parser.add_argument("--outfile", type=Path, help="Output parquet")
    parser.add_argument("--label", type=str, default="d1", help="Label")
    parser.add_argument("--topn", type=int, default=10, help="Top-N labels")
    args = parser.parse_args()
    main(args)
