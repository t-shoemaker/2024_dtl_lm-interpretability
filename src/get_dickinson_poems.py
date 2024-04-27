#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import unicodedata
import time
import re

import requests
from bs4 import BeautifulSoup
import pandas as pd

# Globals for the website
BASE_URL = "https://www.poetryfoundation.org/poets/emily-dickinson"
POEM_TAB = "tab-poems"
POEM_TITLE = "c-feature-hd"
POEM_BODY = "c-feature-bd"


def get_html_data(url):
    """Get HTML data from a URL.

    Parameters
    ----------
    url : str
        The URL to the HTML data

    Returns
    -------
    response : str
        GET response data
    """
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Could not get data from: {url}")

    return response.text


def get_poem(url):
    """Get poem from a URL.

    Parameters
    ----------
    url : str
        The poem URL

    Returns
    -------
    poem : tuple[str, int, str]
        Poem title, number, and text
    """
    # Get the page
    poem_page = get_html_data(url)
    poem_page = BeautifulSoup(poem_page, "lxml")

    # Get the title card
    title = poem_page.find("div", class_=POEM_TITLE).find("h1")
    title = title.get_text(strip=True)
    title = unicodedata.normalize("NFD", title)
    title, poem_number = title.rsplit(" ", 1)

    # Format the poem number
    poem_number = re.sub(r"\D", "", poem_number)
    poem_number = int(poem_number) if poem_number else None

    # Get the stanzas
    body = poem_page.find("div", class_=POEM_BODY)
    body = body.get_text(separator="\n", strip=True)
    body = unicodedata.normalize("NFD", body)

    return title, poem_number, body


def main(args):
    """Run the script."""
    # Make the initial request to the main page
    main_page = get_html_data(BASE_URL)
    poem_block = BeautifulSoup(main_page, "lxml").find("div", id=POEM_TAB)
    poem_links = poem_block.find_all("div", class_="c-invisicard")

    # Set up and output directory for the peoms
    poem_dir = args.datadir / "poems"
    if not poem_dir.exists():
        poem_dir.mkdir(parents=True, exist_ok=True)

    # March through the links
    print("Found", len(poem_links), "poems. Scraping...")
    poems = []
    for idx, link in enumerate(poem_links):
        href = link.find("a").get("href", "")
        title, poem_number, poem = get_poem(href)

        # Save the poem
        fname = str(idx).zfill(2) + ".txt"
        outpath = poem_dir / fname
        with outpath.open("w") as fout:
            print(poem, file=fout)

        # Add its metadata to a running list
        poems.append(
            {"title": title, "number": poem_number, "file": fname}
        )

        # Log, then sleep
        if (idx + 1) % 5 == 0:
            print("Scraped", idx + 1, "poems")

        time.sleep(2)

    # Make a metadata sheet
    poems = pd.DataFrame(poems)
    poems["number"] = pd.to_numeric(poems["number"], errors="coerce")
    poems.to_csv(args.datadir / "metadata.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=Path, help="Where to store files")
    args = parser.parse_args()
    main(args)
