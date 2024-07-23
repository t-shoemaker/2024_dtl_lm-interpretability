#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_perplexity(logits, labels, batch_size):
    """Calculate perplexity.

    Reference: https://stackoverflow.com/a/77433933

    Parameters
    ----------
    logits : torch.Tensor
        Logits from the model
    labels : torch.Tensor
        Token IDs
    batch_size : int
        Size of the batch

    Returns
    -------
    perplexities : torch.Tensor
        Perplexities for every item in the batch
    """
    # Shift and format the logits and labels
    logits = logits[..., :-1, :].contiguous().squeeze()
    labels = labels[..., 1:].contiguous().squeeze()
    logits, labels = logits.view(-1, logits.size(-1)), labels.view(-1)

    # Compute perplexity
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fn(logits, labels).view(batch_size, -1).mean(dim=1)
    perplexities = torch.exp(loss)

    return perplexities


def prepend_target(
    target, tokenizer, model, n_samp=10, n_tokens=1, batch_size=64
):
    """Prepend a target sequence with `n_tokens`, using mean perplexity as an
    optimization.

    Parameters
    ----------
    target : str
        The target sequence
    tokenizer : GPT2TokenizerFast
        GPT-2 tokenizer
    model : GPT2LMHeadModel
        GPT-2 model
    n_samp : int
        Number of prepending candidates
    n_tokens : int
        Number of tokens to prepend
    batch_size : int
        Batch size

    Returns
    -------
    prepended : tuple
        The prepended sequence and its perplexity score
    """
    # Tokenize the input string into a sequence of token IDs and move to device
    seq = tokenizer(target, return_tensors="pt").input_ids.to(DEVICE)
    perplexity = None

    for _ in range(n_tokens):
        # Sample from the vocabulary
        sample = torch.randint(
            0, model.config.vocab_size, (n_samp, 1), device=DEVICE
        )
        inputs = torch.cat((sample, seq.repeat(n_samp, 1)), dim=-1)

        # Set up batches and process
        batch_losses, batch_seq = [], []
        for i in range(0, n_samp, batch_size):
            end = min(i + batch_size, n_samp)
            current_batch_size = end - i
            batch = inputs[i:end]

            # Send the batch to the model
            with torch.no_grad():
                output = model(input_ids=batch, labels=batch)

            # Shift and format the logits and labels
            batch_perplexity = calculate_perplexity(
                output.logits, batch, current_batch_size
            )

            # Add our batches and sequences to the running buffer
            batch_losses.extend(batch_perplexity.tolist())
            batch_seq.extend(batch)

        # Convert our buffer to a tensor and get the sequence with the smallest
        # perplexity
        batch_losses = torch.tensor(batch_losses, device=DEVICE)
        batch_seq = torch.stack(batch_seq)
        min_idx = torch.argmin(batch_losses)
        seq = batch_seq[min_idx].unsqueeze(0)
        perplexity = batch_losses[min_idx]

    decoded = tokenizer.decode(seq.squeeze(), skip_special_tokens=True).strip()
    perplexity = perplexity.item()

    return decoded, perplexity


def main(args):
    """Run the script."""
    ckpt = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt)
    model.eval()
    model.to(DEVICE)

    print(f"Prompt: '{args.prompt}'")
    output, perplexity = prepend_target(
        args.prompt,
        tokenizer,
        model,
        n_samp=args.n_samp,
        n_tokens=args.n_tokens,
        batch_size=args.batch_size,
    )
    print(f"Output: '{output}'")
    print(f"Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepend tokens to a prompt and calculate perplexity."
    )
    parser.add_argument("prompt", type=str, help="The prompt to be prepended.")
    parser.add_argument(
        "--n_samp",
        type=int,
        default=10,
        help="Number of prepending candidates.",
    )
    parser.add_argument(
        "--n_tokens", type=int, default=1, help="Number of tokens to prepend."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for processing."
    )
    args = parser.parse_args()

    main(args)
