---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [remove_cell]
import os
import warnings
from matplotlib.pyplot import rcParams

os.chdir("..")
warnings.filterwarnings("ignore")
rcParams["figure.dpi"] = 150
```

Generative Pre-Trained Transformers (GPT)
=========================================


## Preliminaries

Here are the libraries we will need.

```{code-cell}
from functools import partial

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig, pipeline
import transformer_lens as tl
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
```

Later on, we will use a small dataset of sentence pairs. Let's load them now.

```{code-cell}
pairs = pd.read_parquet("data/datasets/exclamations.parquet")
```

Now: the model. We'll be using GPT-2, a precursor to models like ChatGPT
released in 2019.

```{code-cell}
ckpt = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast = True)
model = AutoModelForCausalLM.from_pretrained(ckpt)
```

Once it's loaded, put the model in evaluation mode. In addition to this step,
we turn off gradient accumulation with a global value. This way we don't need
the context manager syntax.

```{code-cell}
:tags: [output_scroll]
torch.set_grad_enabled(False)
model.eval()
```

A last setup step: GPT-2 didn't have a padding token, which the `transformers`
library requires. You can set one manually like so:

```{code-cell}
tokenizer.pad_token_id = 50256
print("Pad token:", tokenizer.pad_token)
```


## Text Generation

You'll recognize the workflow for text generation: it works much like what we
did with BERT. First, write out a prompt and tokenize it.

```{code-cell}
prompt = "It was the best of times, it was the"
inputs = tokenizer(prompt, return_tensors = "pt")
```

With that done, send the inputs to the model.

```{code-cell}
outputs = model(**inputs)
```

Just as with other inferences created with `transformers`, there are
potentially a variety of outputs available to you. But we'll only focus on
logits. Whereas in logistic regression, logits are log-odds, here they are just
raw scores from the model.

```{code-cell}
:tags: [output_scroll]
outputs.logits
```

Take a look at the shape of these logits. The model has assigned a big tensor
of logits to every token in the prompt. The number of these tokens is the same
as that of the input sequence, and the size of their tensors corresponds to the
total vocabulary size of the model.

```{code-cell}
assert inputs["input_ids"].size(1) == outputs.logits.size(1), "Unmatched size"
assert model.config.vocab_size == outputs.logits.size(2), "Unmatched size"
```

So far, we do not have a newly generated token. Instead, we have next token
information for every token in our input sequence. Take the last of the logit
tensors to get the one that corresponds to the final token in the input
sequence. It's from this tensor that we determine what the next token should
be.

```{code-cell}
last_token_logits = outputs.logits[:, -1, :]
```

To express these logits in terms of probabilities, we must run them through
**softmax**. The formula for this function is below:

$$
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}
$$

Where:

+ $e$ is the base of the natural logarithm
+ $z_i$ is the logit for class $i$ (in the case, every possible token)
+ $\sum_{j=1}^n e^{z_j}$ is the sum of the exponentials of all logits

This equation looks more intimidating than it is. A toy example implements it
below:

```{code-cell}
z = [1.25, -0.3, 0.87]
exponentiated = np.exp(z)
summed = np.sum(exponentiated)

exponentiated / summed
```

Each of these logits is now a probability. The sum of these probabilities will
equal $1$.

```{code-cell}
np.round(np.sum(exponentiated / summed), 1)
```

That said, there's no need for a custom function when `torch` can do it for us.
Below, we run our last token logits through softmax.

```{code-cell}
probs = F.softmax(last_token_logits, dim = -1)
```

Take the highest value to determine the next predicted token.

```{code-cell}
next_token_id = torch.argmax(probs).item()
print(f"Next predicted token: {tokenizer.decode(next_token_id).strip()}")
```

The model's `.generate()` method will do all of the above. It will also
recursively build a new input sequence so that you can have it return multiple
new tokens.

```{code-cell}
outputs = model.generate(**inputs, max_new_tokens = 4)
print(f"Full sequence: {tokenizer.decode(outputs.squeeze())}")
```

Or, just wrap everything in a `pipeline`.

```{code-cell}
:tags: [remove-stderr]
generator = pipeline("text-generation", model = model, tokenizer = tokenizer)
outputs ,= generator(prompt, max_new_tokens = 4)
print(outputs["generated_text"])
```

Supply an argument to `num_return_sequences` to get multiple output sequences.

```{code-cell}
outputs = generator(prompt, max_new_tokens = 4, num_return_sequences = 5)
for sequence in outputs:
    print(sequence["generated_text"])
```


### Sampling strategies

Even with highly predictable sequences (like our prompt), we will not get the
same output every time we generate a sequence. This is because GPT-2 samples
from the softmax probabilities. It's possible to turn that sampling off
altogether, but there are also a number of different ways to perform this
sampling.

**Greedy sampling** isn't really sampling at all. It takes the most likely
token every time. Setting `do_sample` to `False` will cause GPT-2 to use this
strategy. The outputs will be **deterministic**: great for reliable outputs,
bad for scenarios in which you want varied responses.

```{code-cell}
outputs ,= generator(prompt, max_new_tokens = 4, do_sample = False)
print(outputs["generated_text"])
```

In an earlier chapter, we implemented **top-k** sampling. It limits the
sampling pool to only the top `k`-most probable tokens. This makes outputs more
diverse than in greedy sampling, though it requires hard coding a value for
`k`.

```{code-cell}
outputs = generator(
    prompt,
    do_sample = True,
    max_new_tokens = 4,
    top_k = 50,
    num_return_sequences = 5
)
for sequence in outputs:
    print(sequence["generated_text"])
```

Similar to top-k sampling is **top-p**, or **nucleus sampling**. Instead of
fixing the size of the sampling pool to `k` tokens, this strategy considers the
top tokens whose cumulative probability is at least `p`. Again, this requires a
hard-coded value for `p`, but top-p sampling is more adaptive than top-k.

```{code-cell}
outputs = generator(
    prompt,
    do_sample = True,
    max_new_tokens = 4,
    top_p = 0.9,
    num_return_sequences = 5
)
for sequence in outputs:
    print(sequence["generated_text"])
```

Adjust the **temperature** parameter to control the randomness of model
predictions. The value you use for temperature scales the logits before
applying softmax. Lower temperatures $<1$ make the model outputs more
deterministic by sharpening the probability distribution, while higher
temperatures $>1$ make model outputs more random by flattening the probability
distribution.

Low-temperature output looks like this:

```{code-cell}
outputs = generator(
    prompt,
    do_sample = True,
    max_new_tokens = 4,
    temperature = 0.5,
    num_return_sequences = 5
)
for sequence in outputs:
    print(sequence["generated_text"])
```

High-temperature like this:

```{code-cell}
outputs = generator(
    prompt,
    do_sample = True,
    max_new_tokens = 4,
    temperature = 50.0,
    num_return_sequences = 5
)
for sequence in outputs:
    print(sequence["generated_text"])
```

Set temperature to `1` to use logits as they are.

**Beam searching** is the last strategy. It involves tracking multiple possible
generation sequences simultaneously. During the generation process, the model
retains a predetermined number of sequences, or **beams**, based on their
cumulative probabilities; this number is called the **beam width**. The model
iteratively expands each beam with a predicted token and prunes the beams to
retain only the best ones. Finally, the sequence with the highest cumulative
probability is selected as the output.

```{code-cell}
outputs = generator(
    prompt,
    do_sample = True,
    max_new_tokens = 4,
    num_beams = 10,
    num_return_sequences = 5
)
for sequence in outputs:
    print(sequence["generated_text"])
```

The advantage of doing a beam search is that the model can navigate the
probability space to find sequences that may be better overall choices for
output than if it could only construct one sequence on the flight. Its
disadvantage: beam searches are computationally expensive.

Mixing strategies together usually works best. Use a `GenerationConfig` to set
up your pipeline. This will accept several different parameters, including an
**early stopping** value, which will cut generation off in beam-based searches
once `num_beams` candidates have completed.

```{code-cell}
:tags: [remove-stderr]
config = GenerationConfig(
    max_new_tokens = 25,
    do_sample = True, 
    temperature = 1.5,
    top_p = 0.8, 
    num_return_sequences = 5,
)
generator = pipeline(
    "text-generation",
    model = model,
    tokenizer = tokenizer,
    generation_config = config
)
```

Let's generate text with this configuration.

```{code-cell}
outputs = generator(prompt)
for sequence in outputs:
    print(sequence["generated_text"])
```


### Probable sequences

There are of course any number of ways to evaluate the sequences above using
methods developed in literary studies. But we can supplement said methods with
some metrics that express sequences in terms of a model's expectations.

Let's return to the first way we processed our prompt. Sending the token IDs to
the `labels` parameter enables the model to calculate loss.

```{code-cell}
outputs = model(**inputs, labels = inputs["input_ids"])
print(f"Loss: {outputs.loss.item():.4f}")
```

This is **cross-entropy loss**. We saw it in our first language generation
chapter. It measures the difference between the predicted probability
distribution of the next token (according to the logits) and the actual token
(what's in the input sequence). Exponentiate it to express the loss in terms of
**perplexity**.

```{code-cell}
print(f"Perplexity: {torch.exp(outputs.loss).item():.4f}")
```

Recall that perplexity is the average number of guesses the model has to make
to arrive at the full sequence.

One way to think about sequence candidates is to calculate their perplexity.
Below, we generate multiple sequences from our current `pipeline`, store the
text, and send that text back to the model to calculate perplexity. Note that
we need to run sequences in a for loop when sending them to a model, otherwise
the model will calculate loss on the whole batch.

```{code-cell}
outputs = generator(prompt)
sequences = [sequence["generated_text"] for sequence in outputs]
```

Here's the for loop, which we cap off by formatting to a DataFrame:

```{code-cell}
results = {"text": [], "perplexity": []}
for sequence in sequences:
    inputs = tokenizer(sequence, return_tensors = "pt")
    outputs = model(**inputs, labels = inputs["input_ids"])
    perplexity = torch.exp(outputs.loss).item()

    results["text"].append(sequence)
    results["perplexity"].append(perplexity)

results = pd.DataFrame(results)
```

Let's sort by perplexity and see which sequence is best---though, of course,
what best means here is "most probable according to the model."

```{code-cell}
print(results.sort_values("perplexity")["text"].iloc[0])
```

And here's the worst:

```{code-cell}
print(results.sort_values("perplexity", ascending = False)["text"].iloc[0])
```

What if, instead of mean perplexity for a sequence, we wanted token-by-token
perplexity? That would give us average number of guesses the model would need
to make to get the next token in a sequence. The function below will perform
this calculation.

```{code-cell}
def per_token_perplexity(logits, labels):
    """Calculate the perplexity of each token in a sequence.
    
    Reference: https://stackoverflow.com/a/77433933

    Parameters
    ----------
    logits : torch.Tensor
        Sequence logits
    labels : torch.Tensor
        Sequence token IDs

    Returns
    -------
    perplexities : torch.Tensor
        Every token's perplexity
    """
    # Shift the logits and labels by one position so we start from the
    # transition of the first token to the second token
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    # Sequeeze out the batch dimensions
    logits, labels = logits.squeeze(), labels.squeeze()

    # Calculate the cross entropy loss and exponentiate it for token-by-token
    # perplexity
    loss = F.cross_entropy(logits, labels, reduction = "none")
    perplexities = torch.exp(loss)

    return perplexities
```

Let's run it on the full version of original prompt.

```{code-cell}
prompt = "It was the best of times, it was the worst of times."
inputs = tokenizer(prompt, return_tensors = "pt")
outputs = model(**inputs)
perp_token = per_token_perplexity(outputs.logits, inputs["input_ids"])
```

And the results:

```{code-cell}
decoded = [tokenizer.decode(token) for token in inputs["input_ids"].squeeze()]
for idx, (token, perp) in enumerate(zip(decoded[:-1], perp_token)):
    token, next_token = token.strip(), decoded[idx + 1].strip()
    print(f"{token:6}{next_token:6}{perp.item():.2f}")
```

What other famous sentences have this pattern? You could use such a strategy to
answer this question, which may in turn tell you something about what GPT-2 has
absorbed about famous quotes, stock phrases, and cultural memes.


## The Candidate Pool

Let's think now about candidate tokens. Which tokens are likely to be in
consideration when GPT-2 generates text?


### Going forwards

One way to look at this pool would be to look at the logit tensor for the final
token in a sequence. Instead of picking just the maximum value, the function
below selects `k` most likely tokens.

```{code-cell}
def get_top_candidates(logits, k = 5):
    """Get the top `k` most likely tokens from a tensor of logits.

    Parameters
    ----------
    logits : torch.Tensor
        The logit tensor
    k : int
        Number of tokens

    Returns
    -------
    candidates : list[tuple]
        Top `k` candidates and their probability scores
    """
    # Convert the logits to probabilities and squeeze out the batch dimension
    probs = F.softmax(logits, dim = -1).squeeze()
    
    # Select the top `k` candidates. The function below returns the
    # probabilities and the token IDs
    values, indices = torch.topk(probs, k = k)

    # Decode the token IDs. Zip the result up in a list of tuples with the
    # probabilities and return
    decoded = [tokenizer.decode(token) for token in indices]
    candidates = [
        (token, value.item()) for token, value in zip(decoded, values)
    ]

    return candidates
```

One way to use this function would be to run it on complete sentences. For each
token, we can get a list of candidates that the model would have generated.
Often, this will conflict with what an author has written. Below, for example,
we send GPT-2 the first line of Rosmarie Waldrop's poem, "King Lear's Nothing."

```{code-cell}
prompt = "Thrilled by quantity as language."
inputs = tokenizer(prompt, return_tensors = "pt")
outputs = model(**inputs)
```

Now, we iterate through each token ID, get its corresponding logit tensor, and
get the top `k` candidates for the token.

```{code-cell}
:tags: [output_scroll]
decoded = [tokenizer.decode(token) for token in inputs["input_ids"].squeeze()]
for idx in range(len(decoded)):
    # First, get the current token and the next token in the sequence
    token = decoded[idx].strip()
    next_token = decoded[idx + 1].strip() if idx < len(decoded) - 1 else "END"

    # Extract the corresponding logit and calculate the top candidates. Run
    # `repr()` over the tokens to ensure whitespace characters print correctly
    candidates = get_top_candidates(outputs.logits[:, idx, :])
    candidates = [(repr(token), value) for token, value in candidates]
    
    # Build a table and print to screen
    table = tabulate(candidates, headers = ["token", "prob"], showindex = True)
    print(f"{token} -> {next_token}")
    print(table, end = "\n\n")
```

This logic would work for completions as well. Jed Dobson gives the following
two examples in [this article][nb], which demonstrate differences in how the
model responds to gendered pronouns.

[nb]: https://github.com/jeddobson/blackbox-transformers/tree/main

```{code-cell}
prompts = [
    "Dartmouth College, where she graduated last year with a degree in",
    "Dartmouth College, where he graduated last year with a degree in"
]
inputs = tokenizer(prompts, return_tensors = "pt")
outputs = model(**inputs)
```

Let's look at differences across the top-25 candidate completions for these two
prompts.

```{code-cell}
:tags: [output_scroll]
for prompt, logits in zip(prompts, outputs.logits):
    candidates = get_top_candidates(logits[-1, :], k = 25)
    candidates = [(repr(token), value) for token, value in candidates]
    table = tabulate(candidates, headers = ["token", "prob"], showindex = True)
    
    print(prompt)
    print(table, end = "\n\n")
```

To study this further, you might imagine aggregating whole pools of these
candidates for various prompts. How, at scale, do you see the model respond
differently?

:::{tip}
Were you to do this experiment, you might consider [Jaccard similarity][jac],
which compares how similar two sets are by dividing their [intersection][inter]
by their [union][union]. This would give you a metric to represent similarity
in top-`k` token results.

[jac]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
[inter]: https://en.wikipedia.org/wiki/Intersection_(set_theory)
[union]: https://en.wikipedia.org/wiki/Union_(set_theory)
:::


### Going backwards

Here's something more speculative. Earlier we were able to determine the
perplexity of a sequence by having the model calculate loss. What if we used
that functionality to think about prompting in reverse? Could that also tell us
something about what the model "knows"?

Below, the `prepend_prompt()` function takes an input sequence, `target`, and
samples from all possible tokens in GPT-2. Then, it builds a series of
candidate sequences by prepending those samples to the original sequence. With
these candidate sequences built, it runs them through the model, asking the
model to calculate loss along the way. From loss, we can get to perplexity, and
we store that score along with its corresponding candidate sequence. The
selected candidate sequence is the one with the lowest perplexity---that is,
the sequence the model is most likely to guess.

This function also uses recursion to prepend multiple tokens to a sequence.
That requires some logic we haven't covered yet, but see if you can figure out
how it works!

```{code-cell}
def prepend_prompt(
    target, tokenizer, model, n_samp = 10, n_tokens = 1, perplexity = None
):
    """Prepend a target prompt with `n_tokens` that minimize the target's
    perplexity.

    Note: this function is meant for instructional purposes and does not
    leverage batching. It should not be used at scale.
    
    Parameters
    ----------
    target : str
        The target prompt
    tokenizer : GPT2TokenizerFast
        The tokenizer
    model : GPT2LMHeadModel
        The model
    n_samp : int
        Number of candidate tokens to sample
    n_tokens : int
        Number of tokens to prepend to the prompt
    perplexity : None or float
        Full sequence's perplexity

    Returns
    -------
    prepended : tuple[str, float]
        The prepended prompt and its perplexity
    """
    # First, handle the recursive condition: if no more tokens to prepend,
    # return the target string and perplexity
    if n_tokens < 1:
        return target, perplexity.item()

    # Tokenize the target string and sample `n_samp` from the model's
    # vocabulary
    inputs = tokenizer(target, return_tensors = "pt")
    samples = torch.randint(0, model.config.vocab_size, (1, n_samp))

    # Ensure the target string tokens and sampled tokens are on the same
    # device, then concatenate them to a (n_samp, len(inputs) + 1) batch
    inputs = inputs["input_ids"].to(model.device)
    samples = samples.to(model.device)
    batch = torch.cat(
        (samples.reshape(n_samp, 1), inputs.repeat(n_samp, 1)), dim = -1
    )

    # For each candidate sequence in the batch, run it through the model and
    # calculate the perplexity score. Append it to a buffer
    perplexities = []
    for input_ids in batch:
        output = model(input_ids = input_ids, labels = input_ids)
        perp = torch.exp(output.loss)
        perplexities.append((input_ids, perp))

    # Sort the candidate sequences by perplexity, then select the lowest score.
    # Convert the top candidate sequence to a string
    perplexities.sort(key = lambda x: x[1])
    target, perplexity = perplexities[0]
    target = tokenizer.decode(target, clean_up_tokenization_spaces = True)

    # Recurse. Be sure to decrement `n_tokens`!
    return prepend_prompt(
        target, tokenizer, model, n_samp, n_tokens - 1, perplexity
    )
```

Let's see if we can build a for loop header, `for i in range(10):`.

```{code-cell}
:tags: [output_scroll]
prompt = " i in range(10):"
for _ in range(5):
    prepended, perp = prepend_prompt(
        prompt, tokenizer, model, n_samp = 50, n_tokens = 1
    )
    print(f"{prepended} ({perp:.4f})")
```

It doesn't quite get us what we want, but the reason why is probably clear:
`n_samp` controls how many candidate sequences. If the ideal token isn't
sampled from the tokenizer, then we'll never see it. That said, the output
above does get us somewhat in the realm of code-like tokens. This could work.
Coupled with the strategy of sampling from the entire vocabulary, we could very
well get what we want.

This [script][script] performs the same function above but with batching. That
speeds up the process to some extent---enough so that we can try sampling from
the entire model vocabulary. When we do, we'll see that we get to the correct
answer.

[script]: https://github.com/t-shoemaker/2024_dtl_lm-interpretability/blob/main/src/prompt_prepend_batched.py

```sh
$ python src/prompt_prepend_batched.py " pandas as pd" --n_samp 50257 
```
```
Prompt: ' pandas as pd'
Output: 'import pandas as pd'
Perplexity: 27.4596
```

The problem? It takes several minutes to run on a consumer-grade laptop.
Getting access to a GPU would ameliorate that problem to some extent, but even
better would be to come up with a different, and smarter sampling strategy.
Additionally, you might think about what an ideal outcome of prepending should
be. Is minimizing perplexity actually the best goal for exploring the model's
generation space?


## The Circuit View

```{code-cell}
model = tl.HookedTransformer.from_pretrained(
    "gpt2",
    center_unembed = True,
    center_writing_weights = True,
    fold_ln = True,
    refactor_factored_attn_matrices = True
)
```


### Preprocessing

```{code-cell}
def pad_to_same_length(A, B, pad = 50256):
    """Pad two token ID tensors so they are the same length.

    Parameters
    ----------
    A : torch.Tensor
        First tensor
    B : torch.Tensor
        Second tensor
    pad : int
        Padding token ID

    Returns
    -------
    padded : tuple
        Padded tensors
    """
    # Get shapes of both tensors and find the max dimensions
    (A0, A1), (B0, B1) = A.shape, B.shape
    target = (max(A0, B0), max(A1, B1))

    # Pad each tensor to the max dimensions
    A_pad, B_pad = torch.full(target, pad), torch.full(target, pad)
    A_pad[:A0, :A1] = A
    B_pad[:B0, :B1] = B

    return A_pad, B_pad


def filter_padded(A, B, pad = 50256):
    """Filter padded token ID tensors.

    We do this to control for tensor pairs that may differ in length due to
    subword tokenization. 

    Parameters
    ----------
    A : torch.Tensor
        First tensor
    B : torch.Tensor
        Second tensor
    pad : int
        Padding token ID

    Returns
    -------
    filtered : tuple
        Filtered tensors
    """
    # Find tensors that have the same number of padding tokens in the same
    # positions
    same = (A == pad) == (B == pad)

    # Keep only those
    A_filtered, B_filtered = A[same.all(dim = 1)], B[same.all(dim = 1)]

    return A_filtered, B_filtered


def find_variant_pairs(A, B, pad = 50256):
    """Find where two pairs of token ID tensors vary.

    Parameters
    ----------
    A : torch.Tensor
        First tensor
    B : torch.Tensor
        Second tensor
    pad : int
        Padding token ID

    Returns
    -------
    A, B, variants, indices : tuple
        The two tensors, a (n_row, 2) size tensor of token IDs, and another
        (n_row, 1) tensor of the indices where the variants occur
    """
    # Find where the tensors to do not match
    indices = (A != B).nonzero()

    # Drop multi-token variants
    unique, counts = torch.unique(indices, return_counts = True)
    mask = torch.isin(indices[:, 0], unique[(counts == 1) | (counts != pad)])
    indices = indices[mask]

    # Compile a variants tensor
    variants = []
    for doc_id, token_id in indices:
        A_id = A[doc_id, token_id].item()
        B_id = B[doc_id, token_id].item()
        variants.append([A_id, B_id])
    variants = torch.tensor(variants, device = model.cfg.device)

    return A, B, variants, indices[:, 1]
```

We separate clean from corrupted sentences and tokenize them.

```{code-cell}
clean = model.to_tokens(pairs["clean"])
corrupted = model.to_tokens(pairs["corrupt"])
```

Now, we need to pad our tokens IDs. Then, we must ensure that each sentence
pair is the same length. Our calculations will be incorrect if we do not do
this.

```{code-cell}
clean, corrupted = pad_to_same_length(clean, corrupted)
clean, corrupted = filter_padded(clean, corrupted)
```

Finally, we identify which token IDs differ across the sentence pairs, and
where those variations appear.

```{code-cell}
clean, corrupted, variants, indices = find_variant_pairs(clean, corrupted)
```

Here are the variant pairs. The token IDs are for "." and "!", respectively.

```{code-cell}
:tags: [output_scroll]
variants
```

And here is where these variants appear in the token tensors.

```{code-cell}
:tags: [output_scroll]
indices
```


### Activation patching

```{code-cell}
clean_logits, clean_cache = model.run_with_cache(clean)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted)
```

These caches contain all the different states of the model for our two
collections of sentence pairs. For example, we can retrieve the output of the
attention heads at the fourth layer in the model for clean sentences.

```{code-cell}}
:tags: [output_scroll]
name = tl.utils.get_act_name("attn_out", 4)
clean_cache[name]
```

```{code-cell}
def logit_diff(logits, variants = variants, indices = indices, dim = 0):
    """Find the difference between two logit tensors.

    Parameters
    ----------
    logits : torch.Tensor
        Logit tensors from the model
    variants : torch.Tensor
        A (n_row, 2) tensor of token IDs with the clean and corrupted tokens
    indices : torch.Tensor
        A (n_row, 1) tensor of locations where tokens differ

    Returns
    -------
    difference : float
        The difference between the clean and corrupted versions of the tensors
    """
    # If we are dealing with batched tensors, select the index position for the
    # flipped token
    if len(logits.shape) == 3:
        flipped = torch.arange(len(indices))
        logits = logits[flipped, indices]

    # Get the logits for the clean tokens. Note the indexing along the first
    # dimension of `variants`. That selects the uncorrupted token IDs
    correct = logits.gather(1, variants[:, 0].unsqueeze(1))

    # Get the logits for the corrupted tokens. This selects corrupted token IDs
    # from variants
    incorrect = logits.gather(1, variants[:, 1].unsqueeze(1))

    # Subtract incorrect logits from the correct ones and take the mean
    difference = (correct - incorrect).mean(dim = dim)

    return difference
```

We define two baselines to normalize logit differences as the model runs. 

```{code-cell}
CLEAN_BASELINE = logit_diff(clean_logits, variants, indices).item()
CORRUPT_BASELINE = logit_diff(corrupted_logits, variants, indices).item()
```

Now, we define a function to compute a metric for the model. This is a loss
function, which the patching process uses to determine whether a permutation at
a given layer increases/decreases the likelihood of the correct outcome.

```{code-cell}
def metric(logits, variants = variants, indices = indices):
    """Compute logit difference and normalize the results.

    Parameters
    ----------
    logits : torch.Tensor
        Logit tensors from the model
    variants : torch.Tensor
        A (n_row, 2) tensor of token IDs with the clean and corrupted tokens
    indices : torch.Tensor
        A (n_row, 1) tensor of locations where tokens differ

    Returns
    -------
    difference : float
        The difference between the clean and corrupted versions of the tensors
    """
    difference = logit_diff(logits, variants, indices)
    norm_by = CLEAN_BASELINE - CORRUPT_BASELINE
    difference = (difference - CORRUPT_BASELINE) / norm_by

    return difference
```

Time to run activation patching. This compares the corrupted tokens to the
clean cache's attention scores.

```{code-cell}
:tags: [remove-stderr]
patched = tl.patching.get_act_patch_attn_out(
    model, corrupted, clean_cache, metric
)
results = patched.cpu().numpy()
```

Let's look at a heatmap of the results. That will show where, at a given
position in our sentence pairs, making changes at a layer increases the
likelihood of the correct outcome or decreases that likelihood. Positive
numbers are increases, negative are decreases.

```{code-cell}
plt.figure(figsize = (9, 4))
g = sns.heatmap(
    results,
    cmap = "crest",
    annot = True,
    fmt = ".2f",
    robust = True,
    mask = results == 0
)
g.set(title = "Patching outcomes", xlabel = "Token position", ylabel = "Layer")
plt.show()
```

From the looks of this heatmap, activation scores in the tenth layer of the
model are most sensitive to changes between clean and corrupted tokens.


### Steering the model

```{code-cell}
:tags: [output_scroll]
model.to("cpu")
```

```{code-cell}
def make_steering_vector(clean, corrupt, name = None, model = model):
    """Make a steering vector from two tokens.

    Parameters
    ----------
    clean : str
        Clean token
    corrupt : str
        Corrupt token
    name : str
        Name key for the model cache
    model : tl.HookedTransformer
        A hooked model

    Returns
    -------
    steering_vector : torch.Tensor
        A steering vector of size (1, n_dim)
    """
    # Generate vectors for each token
    vectors = []
    for tok in (clean, corrupt):
        _, cache = model.run_with_cache(tok)

        # Extract the activations at a specified layer
        vectors.append(cache[name])

    # Unpack the tokens and subtract the second from the first to define a
    # relationship between the two
    clean, corrupt = vectors
    steering_vector = clean - corrupt

    # Return the results
    return steering_vector[:, 1, :]
```

```{code-cell}
layer_id = 10
name = tl.utils.get_act_name("attn_out", layer_id)
steering_vector = make_steering_vector(".", "!", name = name)
```

Finally, write a **hook** that will modify the activations at our specified
layer using the steering vector. During a forward pass, the model calls this
function when it reaches the layer.

```{code-cell}
def steer(activations, hook, steering_vector = None, coef = 1.0):
    """Modify activations with a steering vector.

    Parameters
    ----------
    activations : torch.Tensor
        Model activations
    hook : Any
        Required for compatability with PyTorch but unused
    steering_vector : torch.Tensor
        The steering vector
    coef : float
        A coefficient for scaling the steering vector

    Returns
    -------
    steered : torch.Tensor
        Steered activations
    """
    if steering_vector is None:
        return activations

    return activations + steering_vector * coef
```

Now, we define a **partial** function for the hook. All this does is set up
some default arguments; the only arguments left to take are for activations and
the hook, which the model takes care of itself. Using a negative coefficient
will steer the model towards corrupted output.

```{code-cell}
hook = partial(steer, steering_vector = steering_vector, coef = -3.5)
```

Time to generate text. Below, we set up a for loop that covers two routines:
hooking the model and running it without the hook. First, we'll run this
without any sampling.

```{code-cell}
prompt = "The sky is blue"
for do_hook in (True, False):
    # First, reset all hooks. Then, if we're running with a hook, register it
    model.reset_hooks()
    if do_hook:
        model.add_hook(name = name, hook = hook)

    # Generate text. Note that the hooked model has slightly different
    # parameters for its `.generate()` method
    outputs = model.generate(
        prompt,
        max_new_tokens = 1,
        do_sample = False,
        stop_at_eos = True,
        eos_token_id = model.tokenizer.eos_token_id,
        verbose = False
    )

    # What do we get?
    print(f"Hooked: {do_hook}")
    print(f"Output: {outputs}\n-------")
```

Promising... But what about with sampling?

```{code-cell}
model.reset_hooks()
model.add_hook(name = name, hook = hook)

outputs = model.generate(
    prompt,
    max_new_tokens = 50,
    do_sample = True,
    temperature = 0.75,
    top_p = 0.8,
    freq_penalty = 2,
    stop_at_eos = True,
    eos_token_id = model.tokenizer.eos_token_id,
    verbose = False
)
print(outputs)
```

Even more promising. Let's crank the effect of the steering vector way up.

```{code-cell}
hook = partial(steer, steering_vector = steering_vector, coef = -8.5)
model.reset_hooks()
model.add_hook(name = name, hook = hook)
```

...and conclude with a final prompt.

```{code-cell}
prompt = "I had an okay time"
outputs = model.generate(
    prompt,
    max_new_tokens = 50,
    do_sample = True,
    temperature = 0.75,
    top_p = 0.8,
    freq_penalty = 2,
    stop_at_eos = True,
    eos_token_id = model.tokenizer.eos_token_id,
    verbose = False
)
print(outputs)
```

Not only do we see more exclamation marks, but content changes as well.
