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
from matplotlib.pyplot import rcParams

os.chdir("..")
rcParams["figure.dpi"] = 150
```

Data Analysis in Python
=======================

This chapter will show you the basics of analyzing data in Python. We will load
text files into memory, align them with corresponding metadata, and produce
information about their contents. Also covered: preparing text for numerical
operations and graphing data.

+ **Data**: Melanie Walsh’s [corpus][data] of ~380 obituaries from the _New
  York Times_

[data]: https://melaniewalsh.github.io/Intro-Cultural-Analytics/00-Datasets/00-Datasets.html#politics-history


## Preliminaries

The packages we'll need today will help us load text files (`pathlib`), process
them into discrete tokens (`nltk`), conduct data analysis about those tokens
(`numpy`, `pandas`), and plot the results (`seaborn`, `matplotlib`).

```{code-cell}
from pathlib import Path
import re

import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```


## Loading a Corpus

The obituaries are stored in individual plain text files at the location below.
We wrap this file path in a `Path` object to make interacting with our
computers' file systems more streamlined.

```{code-cell}
datadir = Path("data/nyt_obituaries/texts")
```

Use a **glob** pattern to retrieve paths to `.txt` files. The output of the
`.glob()` method is a generator, so convert it to a list.

```{code-cell}
paths = list(datadir.glob("*.txt"))
print(paths[:5])
```

As we did in the last chapter, we can load one of these files. Note the slight
difference in syntax when using `Path`.

```{code-cell}
:tags: [output_scroll]
random_path = np.random.choice(paths)
with random_path.open("r") as fin:
    doc = fin.read()
    print(doc)
```

It will make our lives easier to define a function that loads all files at
once. That way we only have to call that function, rather than rewriting some
loading code every time we want files. Here is what the `load_corpus()`
function does below:

1. Steps through each path in `paths`
2. Opens the file and appends it to a list

```{code-cell}
def load_corpus(paths):
    """Load a corpus from paths.

    Parameters
    ----------
    paths : list[Path]
        A list of paths

    Returns
    -------
    corpus : list[str]
        The corpus
    """
    # Initialize an empty list to store the corpus
    corpus = []
    
    # March through each path, open the file, and load it
    for path in paths:
        with path.open("r") as fin:
            doc = fin.read()
            # Then add the file to the list
            corpus.append(doc)

    # Return the result: a list of strings, where each string is the contents
    # of a file
    return corpus
```

With this function defined, we load our files.

```{code-cell}
corpus = load_corpus(paths)
print("Size of the corpus:", len(corpus), "files.")
```


## Working with Tabular Data

Note however that the file names do not tell us the title of these poems:

```{code-cell}
print(random_path.name)
```

Were we to run analyses on these documents, we would have no guide telling us
which data is about which document. This is where metadata comes in. Often when
working with text data, you will find information about the contents of a
corpus stored separately from the data itself. Part of your workflow will
require aligning corpus contents with this metadata.


### Loading tabular data

In our case, metadata is stored in a comma-separated (CSV) file, a plain text
format for **tabular data**. Tabular data arranges information in columns and
rows, just like a spreadsheet. The `pandas` package helps us work with this
kind of data. When we load it into Python, we create a **DataFrame**. Just as
with a spreadsheet, a DataFrame has columns and rows. But it also offers a huge
amount of functionality for working with its contents.

Below, we load our metadata.

```{code-cell}
manifest = pd.read_csv("data/nyt_obituaries/metadata.csv")
```

Here is a high-level overview of the metadata. It shows the columns and their
names, the number of observations in each column that contain values, and the
datatype of these columns.

```{code-cell}
manifest.info()
```

Just want to know the columns? Use the `.columns` attribute.

```{code-cell}
manifest.columns
```

Use the `.head()` method to look at the first few rows of the data.

```{code-cell}
manifest.head()
```

Note the `file` column. Values stored there correspond to the names of files in
the data directory. If we use `file` as a guide to construct a list of paths,
the order of the files in `corpus` will be the same as the order in the
DataFrame.


### Indexing by column

Accessing values in `file` requires us to index the DataFrame. In `pandas`, we
use bracket notation in conjunction with a column's name to index that column.

```{code-cell}
:tags: [output_scroll]
manifest["file"]
```

There are several ways to index rows, which we discuss below. But the simplest
involves treating an indexed column like a list.

```{code-cell}
manifest["file"][10]
```

The above hints at what we do next: we use a list comprehension to iterate
through each value in `file` and combine it with the data directory path. Note
this time that we do not need to use a glob pattern, since we build the final
path directly from the value in `file` using the `/` operator.

```{code-cell}
ordered_paths = [datadir / fname for fname in manifest["file"]]
```

Now, when we load our corpus, it will be aligned to the metadata's order.

```{code-cell}
corpus = load_corpus(ordered_paths)
```

From here, we could go about our analysis. But that would involve jumping
across two different objects, `manifest` and `corpus`. This is a pain, so we
will create a new column in our metadata sheet and assign the contents of our
corpus to it.

```{code-cell}
manifest["text"] = corpus.copy()
```

Under the hood, every column in a DataFrame is a **Series**. A DataFrame, in
other words, is a collection of Series objects. The latter have much of the
same functionality as the former, but DataFrames provide us with the ability to
do more faceted indexing and global analyses.

For example, now that our corpus contents are stored in the `text` column, we
can index that data alongside other information in the DataFrame. To do so, use
a list of column names.

```{code-cell}
manifest[["name", "text"]].head()
```


### Indexing by row

Indexing by rows is more complicated than indexing by columns. This is because
a DataFrame index serves three important roles:

1. As metadata that provides more context about a dataset
2. As a method of data alignment
3. As a convenience function for subsetting data

Use the `.index` attribute to access the values of a DataFrame index. These
values can be numbers, strings, dates, or other values.

```{code-cell}
manifest.index
```

Like tuples, indexes are immutable. But you can change the index of a
DataFrame. Below, we set the index to `title`, using `inplace = True` so we do
not need to reassign the DataFrame back to the same variable.

```{code-cell}
manifest.set_index("name", inplace = True)
manifest.head()
```

There are three ways to index by row:

1. By integer position
2. By label/name
3. By a condition

Indexing by **integer position** works with the `.iloc` property.

```{code-cell}
manifest.iloc[45]
```

Use a sequence of values to return multiple rows:

```{code-cell}
manifest.iloc[[2, 4, 6, 8, 10]]
```

Or send a slice. Here, the first five rows:

```{code-cell}
manifest.iloc[0:5]
```

Here, every tenth row:

```{code-cell}
:tags: [output_scroll]
manifest.iloc[::10]
```

Indexing by **label** works with `.loc`.

```{code-cell}
name = "John Dewey"
manifest.loc[name]
```

Use it conjunction with a column name to access the value in a cell.

```{code-cell}
:tags: [output_scroll]
print(manifest.loc[name, "text"])
````

Or send it a sequence of labels:

```{code-cell}
names = ["John Dewey", "Lucille Ball"]
manifest.loc[names]
```

Finally, there is indexing by **condition**. This works by evaluating a
condition and returning a Series of Boolean values. It is the most powerful
method of indexing in Pandas by far.

Below, we find all poems with names that start with "S". Use the `.str`
attribute of an index of strings to accomplish this.

```{code-cell}
:tags: [output_scroll]
manifest.index.str.startswith("S")
```

See the Boolean values? Let's assign the output of the above to a mask
variable, with which we will index the DataFrame.

```{code-cell}
mask = manifest.index.str.startswith("S")
manifest.loc[mask]
```

What makes indexing by condition so powerful is that it generalizes to other
data in the DataFrame, not just the index. Let's reset the index and explore
this a little.

```{code-cell}
manifest.reset_index(inplace = True)
manifest.head()
```

Below, we find obituaries that contain the string "musician" (note that this
will return the plural as well).

```{code-cell}
manifest.loc[manifest["text"].str.contains("musician"), "name"]
```

Here, something more complicated: documents that contain the string "paintings"
with file names above `09.txt`.

```{code-cell}
mask = (manifest["text"].str.contains("Bird")) & \
    (manifest["file"].str.startswith("00") == False)
manifest.loc[mask, "name"]
```


## Preparing for Data Analysis

With the basics of indexing done, we will prepare to analyze the corpus. This
will involve two steps. First, we preprocess the raw text in the `text` column
of our DataFrame. Then, we define some plotting functions to graph the results
of our analysis.


### Preprocessing

As we saw in the last chapter, operations like counting require texts to be
processed in special ways. This includes changing the case of texts, breaking
texts into lists of tokens, and so forth.

Previously, we used a simple heuristic to tokenize text: split the text stream
on whitespace characters.

```{code-cell}
:tags: [output_scroll]
example = manifest.loc[manifest["name"] == "Ada Lovelace", "text"].item()
print(example.split())
```

The problem with this is that it cannot handle punctuation that is directly
attached to the preceding characters, as in the case of periods, commas, etc.
To get around this, we use a more sophisticated tokenizer from the `nltk`
package, which is based on a series of regexes.

```{code-cell}
print(nltk.wordpunct_tokenize(example))
```

Below, we incorporate this tokenizer into a preprocessing function that
performs the following steps:

1. Change the string to lowercase
2. Tokenize the string into lists of tokens
3. Optionally, create multi-gram token sequences (more on this next week)

```{code-cell}
def preprocess(doc, ngram = 1):
    """Preprocess a document.

    Parameters
    ----------
    doc : str
        The document to preprocess
    ngram : int
        How many n-grams to break the document into
    
    Returns
    -------
    tokens : list
        Tokenized document
    """
    # First, change the case of the words to lowercase
    doc = doc.lower()

    # Tokenize the string. Optionally, make 2-gram (or more) sequences from
    # those tokens
    tokens = nltk.wordpunct_tokenize(doc)
    if ngram > 1:
        tokens = list(nltk.ngrams(tokens, ngram))
    
    return tokens
```

With our function defined, we preprocess the corpus documents.

```{code-cell}
cleaned = [preprocess(doc) for doc in manifest["text"]]
```

Then we get rid of punctuation and numbers with a regex substitution. Note that
this is a two-step processed: first we remove anything that isn't an alphabetic
character, then we filter out empty strings in the sublists.

```{code-cell}
cleaned = [[re.sub(r"[^a-zA-Z]", "", tok) for tok in doc] for doc in cleaned]
cleaned = [[tok for tok in doc if tok] for doc in cleaned]

print(cleaned[0])
```

Finally, we assign to our DataFrame.

```{code-cell}
manifest["tokens"] = cleaned.copy()
```

### Plotting

A last step before analyzing these tokens: defining a function to plot our
results with a **histogram**. We use `seaborn` for this. It has a simple
interface that integrates directly with DataFrames.

```{code-cell}
def plot_metrics(data, variable, title = "", xlabel = "", figsize = (15, 5)):
    """Plot metrics with a histogram.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot
    variable : str
        Which variable to plot
    title : str
        Plot title
    xlabel : str
        Label of the X axis
    figsize : tuple
        Size of the figure
    """
    # First, check whether the variable we want to plot is in the DataFrame
    if variable not in data.columns:
        raise ValueError(variable, "not in data")

    # Create a figure with an axis
    fig, ax = plt.subplots(figsize = figsize)

    # Put a graph on that axis, then set some features
    g = sns.histplot(data = data, x = variable)
    g.set(title = title, xlabel = xlabel, ylabel = "Count");
```


## Data Analysis

Time to look at our data. Many of these operations will rely on the `.apply()`
method. You can think of this method like a `for` loop: it applies some
function to every element along an **axis** in the DataFrame. Axis `0` is the
column axis, while `1` is the row axis. This feels somewhat backwards, but
setting `axis = 0` applies a function to all rows under a column; `axis = 1`
applies a function to all columns across a row.


### Document metrics

First, some simple document metrics. Below, we calculate the **number of
tokens** in a document, as expressed in this notation:

$$
T(i) = \Sigma_{j=1}^{m_i}1
$$

Where:

+ $T(i)$ is the total number of tokens for the $i$-th document
+ $m_i$ represents the total number of tokens in every token list
+ Each token $j$ in document $i$ is counted once, indicated by $1$

In code, `len()` will handle this easily.

```{code-cell}
manifest["num_tokens"] = manifest["tokens"].apply(len)
plot_metrics(manifest, "num_tokens", title = "Token counts", xlabel = "Tokens")
```

The **number of types** is the number of unique tokens in a document. We
calculate it with:

$$
K(i) = \Sigma_{j \in J}1
$$

Where:

+ $K(i)$ is the total number of types for the $i$-th document
+ $j \in J$ represents each token $j$ for $J$ unique tokens
+ Each token $j$ in $J$ is counted once, indicated by $1$

To implement in code, we take advantage of a feature in the `.apply()` method:
its outputs can be directed to another `.apply()` call, or **chained**.

```{code-cell}
manifest["num_types"] = manifest["tokens"].apply(np.unique).apply(len)
plot_metrics(manifest, "num_tokens", title = "Type counts", xlabel = "Types")
```

The **type-token** ratio is a measure of lexical diversity.

$$
TTR(i) = \frac{K(i)}{T(i)}
$$

In other words, for document $i$ it is the number of types $K(i)$ divided by
the number of tokens $T(i)$.

```{code-cell}
manifest["ttr"] = manifest["num_types"] / manifest["num_tokens"]
plot_metrics(manifest, "ttr", title = "Type-token ratio", xlabel = "TTR")
```

Use the `.nlargest()` method to find the document with the highest type-token
ratio.

```{code-cell}
manifest.nlargest(n = 1, columns = "ttr")
```

And `.nsmallest()` will return the lowest one:

```{code-cell}
manifest.nsmallest(n = 1, columns = "ttr")
```

Finally, a global view of these three metrics using `.describe()`:

```{code-cell}
manifest[["num_tokens", "num_types", "ttr"]].describe()
```


### Token metrics

Now, tokens. Next week we will use a special data structure, the
**document-term matrix** to make working with token data easier, but base
functionality in `pandas` will suffice for now. Using `.explode()` breaks token
lists into individual rows.

```{code-cell}
manifest = manifest.explode("tokens")
```

That greatly lengthens the DataFrame. You will often hear of data scientists
speak of **long** and **wide** data. That refers to tabular data that has many
observations relative to variables (long) or vice versa (wide).

The `.shape` attribute stores information about the number of rows and columns.

```{code-cell}
num_rows, num_cols = manifest.shape
print(f"DataFrame dimensions: ({num_rows:,} x {num_cols})")
```

Use `.value_counts()` to count observations in a column. We assign the result
to a new variable, convert it to a DataFrame, and then use `.sort_values()` to
order them in descending order.

```{code-cell}
token_freq = manifest["tokens"].value_counts()
token_freq = pd.DataFrame(token_freq).reset_index()
token_freq.sort_values(by = "count", ascending = False, inplace = True)
```

Tokens with the highest frequency:

```{code-cell}
token_freq.head(10)
```

And the lowest:

```{code-cell}
token_freq.tail(10)
```

Though there are in fact many tokens that only occur once in the data. We refer
to these as **hapax legomena** (Greek for "only said once"). How many are
there?

```{code-cell}
hapaxes = token_freq[token_freq["count"] == 1]
print(f"Number of hapaxes: {len(hapaxes):,}")
```

A broader look at token counts will situate hapaxes. Below, we plot the 1,000
most frequent tokens.


```{code-cell}
N = 1000
fig, ax = plt.subplots(figsize = (15, 5))
g = sns.scatterplot(
    data = token_freq[:N], x = "tokens", y = "count", ax = ax
)
g.set(xlabel = "Tokens", ylabel = "Token counts", title = f"Top {N:,} Tokens")
plt.xticks(rotation = 90, ticks = range(0, N, 25))
plt.show()
```

Even in the top 1,000 tokens, it's evident that there is an extremely **long
tail** in the count data. More, even at the highest counts there are big jumps
between the most frequent token, the second one, the third, and so on.

Plotting a larger sample will show the same pattern. Below, we sample 10,000
tokens randomly.

```{code-cell}
N = 10_000
sampled = token_freq.sample(N, replace = False)
sampled.sort_values("count", ascending = False, inplace = True)
```

Now we plot on a line plot.

```{code-cell}
fig, ax = plt.subplots(figsize = (15, 5))
g = sns.lineplot(sampled, x = "tokens", y = "count")
g.set(
    xlabel = "Tokens", ylabel = "Count", title = f"Sampled (N = {N:,}) Tokens"
)
plt.xticks(rotation = 90, ticks = range(0, N, 500))
plt.show()
```

What is this telling us? Our token distribution is **Zipfian**. The $n$-th
value of a token is inversely proportional to its position $n$. Or, put another
way, the most common token in the data occurs twice as often as the next most
common token, three times as often as the third most common token, and so on.

Importantly, the most frequent tokens in this data are **deictic** words: words
like "and," "the," etc. These words are the very sinew of language, and yet
they're so redundant and so context-dependent that it's difficult to get a
sense of what they mean. A great number of language models start with this very
problem---including Claude Shannon's mathematical theory of communication, the
subject of our next chapter.

