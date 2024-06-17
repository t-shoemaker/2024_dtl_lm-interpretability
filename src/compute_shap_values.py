#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import torch
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
import shap

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main(args):
    # Load the blurbs and sample them
    blurbs = pd.read_parquet(args.infile)
    sample = blurbs.sample(args.sample_size)

    # Initialize a pipeline
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        device=DEVICE,
    )

    # Set up an explainer and get SHAP values
    explainer = shap.Explainer(
        pipe, output_names=list(model.config.id2label.values()), seed=357
    )
    explained = explainer(sample["text"].tolist(), max_evals=500)

    # Reformat SHAP values as a DataFrame
    shap_values = []
    for idx, entry in enumerate(explained):
        # Get the values for each class
        df = pd.DataFrame(entry.values, columns=explained.output_names)

        # Yank the tokens and strip whitespace
        tokens = [token.strip() for token in entry.data]

        # Create document and token indices
        doc_idx = [idx for token in tokens]
        token_idx = [i for i, token in enumerate(tokens)]

        # Set an index for the DataFrame
        df.index = pd.MultiIndex.from_arrays(
            [doc_idx, token_idx, tokens],
            names=["document_id", "token_id", "text"],
        )
        shap_values.append(df)

    # Form a big DataFrame, then drop blank tokens. These correspond to BOS/EOS
    # tags, which we ignore for this analysis
    shap_values = pd.concat(shap_values)
    shap_values = shap_values[shap_values.index.get_level_values(2) != ""]

    # Reformat base values as a DataFrame
    base_values = pd.DataFrame(
        explained.base_values, columns=explained.output_names
    )
    base_values.index.name = "document_id"

    # Save
    shap_values.to_parquet(args.shap)
    base_values.to_parquet(args.base)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path, help="Input DataFrame with text")
    parser.add_argument("--shap", type=Path, help="SHAP values DataFrame")
    parser.add_argument("--base", type=Path, help="Base values DataFrame")
    parser.add_argument("--model", type=Path, help="Local model to use")
    parser.add_argument(
        "--sample_size", type=int, default=1000, help="Num. samples"
    )
    args = parser.parse_args()
    main(args)
