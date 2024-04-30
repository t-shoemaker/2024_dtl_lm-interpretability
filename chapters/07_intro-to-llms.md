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

Large Language Models: An Introduction
======================================

## Using a Pretrained Model


### Hugging Face


### Loading a model

```py
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
bert = AutoModel.from_pretrained(checkpoint)
```


## Subword Tokenization

```py
sentence = "Then I tried to find some way of embracing my mother's ghost."
```

```py
inputs = tokenizer(
    sentence, return_tensors = "pt", return_attention_mask = True
)
```

### Input IDs

```py
inputs["input_ids"]
```

```py
tokenizer.decode(5745)
```

Two special tokens, `[CLS]` and `[SEP]`:

```py
tokenizer.decode([101, 102])
```

### Token type IDs

```py
inputs["token_type_ids"]
```

### Attention mask

```py
inputs["attention_mask"]
```


## Components of the Transformer

```py
:tags: [output_scroll]
print(bert)
```


### Input embeddings

```py
embeddings = bert.embeddings.word_embeddings(tokens["input_ids"])
```

Squeeze for one batch

```py
embeddings = embeddings.squeeze(0)
```


### Attention

```py
def calculate_attention(embeddings):
    """Calculate attention across word embeddings.

    Parameters
    ----------
    embeddings : torch.Tensor
        Word embeddings of (n_tokens, n_dimensions)

    Returns
    -------
    weighted : torch.Tensor
        Word embeddings weighted by attention
    """
    # Find the shape of the embeddings
    n_tokens, n_dimensions = embeddings.shape

    # Make the Key, Query, and Value matrices, all of which begin as copies of
    # the original embeddings
    K = Q = V = embeddings

    # Perform matrix multiplication to query keys (dot product of the i-th 
    # and j-th rows of `Q`, `K`). Note the transpose of `K`
    scores = torch.matmul(Q, K.T)

    # Normalize the scores with the square root of `n_dimensions`. This reduces
    # the magnitude of the scores, thereby preventing them from becoming too
    # large (which would in turn create vanishingly small gradients during back
    # propagation)
    norm_by = torch.sqrt(torch.tensor(n_dimensions, dtype = torch.float32))
    scores = scores / norm_by

    # Compute softmax to convert attention scores into probabilities. Every row
    # in `probs` is a probability distribution across every token in the model
    probs = F.softmax(scores, dim = -1)

    # Perform a final matrix multiplication between `probs` and `V`. Here,
    # `probs` acts as a set of weights by which to modify the original
    # embeddings. Matrix multiplication will aggregate all values in `V`,
    # producing a weighted sum
    weighted = torch.matmul(probs, V)

    return weighted
```

```py
attention_scores = calculate_attention(embeddings)
```


### Linear transform

```py
linear_layer = nn.Linear(in_features = 768, out_features = 3072, bias = True)
transformed = linear_layer(attention_scores)
```


### Activation layer

```py
activation_layer = nn.ReLU()
activations = activation_layer(transformed)
```


## Running the Model

```py
outputs = bert(**inputs)
```


### Which layer?


## Excerise

Compare the word "sign" in Saussure to the base embedding in the model.
