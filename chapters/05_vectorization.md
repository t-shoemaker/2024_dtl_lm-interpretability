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

Vectorization
=============

## Preliminaries

We will need the following libraries:

```{code-cell}
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
```

Our corpus files are stored in a DataFrame alongside other metadata.

```{code-cell}
data = pd.read_parquet("data/nyt_obituaries.parquet")
```

Now, a quick snapshot of the contents. First, years covered:

```{code-cell}
fig, ax = plt.subplots(figsize = (15, 5))
g = sns.histplot(data = data, x = "year", bins = 30)
g.set(title = "Obituaries per year", xlabel = "Year", ylabel = "Count");
```

Some example text:

```{code-cell}
sampled = data.sample(n = 3)
for row in sampled.iterrows():
    _, row = row
    tokens = row["text"].split()
    print(f"{row['name']}: {tokens[:5]}")
```


## The Document-Term Matrix

```{code-cell}
tf_vectorizer = CountVectorizer()
dtm = tf_vectorizer.fit_transform(data["text"])
```

`CountVectorizer` returns a **sparse matrix**, or a matrix comprised mostly of
zeros.

```{code-cell}
dtm = pd.DataFrame(
    dtm.toarray(),
    columns = tf_vectorizer.get_feature_names_out(),
    index = data["name"]
)
```

```{code-cell}
def plot_metrics(data, col, title = "", xlabel = ""):
    """Plot metrics with a histogram.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot
    col : str
        Which column to plot
    title : str
        Plot title
    xlabel : str
        Label of the X axis
    """
    if col not in data.columns:
        raise ValueError(col, "not in data")

    fig, ax = plt.subplots(figsize = (15, 5))
    g = sns.histplot(data = data, x = col)
    g.set(title = title, xlabel = xlabel, ylabel = "Count");
```

### Document metrics

First, some simple document metrics. Below, we calculate the **number of
terms** in a document, as expressed in this notation:

$$
T(i) = \Sigma_{j=1}^{n}D_{ij}
$$

Where the total terms for each document $i$ in the DTM $D$ is the sum of the
frequency of each term $j$ for $n$ total terms.

```{code-cell}
data["num_terms"] = np.sum(dtm, axis = 1).values
plot_metrics(data, "num_terms", title = "Term counts", xlabel = "Terms")
```

The **number of types** is the number of unique terms in a document. We
calculate it with:

$$
K(i) = \Sigma_{j=1}^{n}1(D_{ij} > 0)
$$

For every document $i$ in DTM $D$, each term $j$ is counted as $1$ if it is
above zero, $1(D_{ij} > 0)$. The type frequency is the summation of those
counts

```{code-cell}
data["num_types"] = np.count_nonzero(dtm, axis = 1)
plot_metrics(data, "num_types", title = "Type counts", xlabel = "Types")
```

The **type-token ratio** is a measure of lexical diversity.

$$
TTR(i) = \frac{K(i)}{T(i)}
$$

In other words, for document $i$ it is the number of types $K(i)$ divided by
the number of terms $T(i)$.

```{code-cell}
data["ttr"] = data["num_types"] / data["num_terms"]
plot_metrics(data, "ttr", title = "Type-token ratio", xlabel = "TTR")
```

Which document has the highest type-token ratio?

```{code-cell}
data.nlargest(n = 1, columns = "ttr")
```


### Term metrics

On to terms. Term frequency is a summation along the column axis of the DTM:

$$
TF(j) = \Sigma_{i=1}^{m}D_{ij}
$$

The term frequency for term $j$ in the DTM $D$ is the sum of its frequency
in document $i$ for $m$ total documents.

```{code-cell}
term_freq = pd.DataFrame(dtm.sum(axis = 0), columns = ["count"])
term_freq.describe()
```

Terms with the highest frequency:

```{code-cell}
term_freq.sort_values("count", ascending = False, inplace = True)
term_freq.head(10)
```

And the lowest:

```{code-cell}
term_freq.tail(10)
```

Though there are in fact many tokens that only occur once in our DTM. We refer
to them as **hapax legomena** (Greek for "only said once"). How many are in the
corpus altogether?

```{code-cell}
hapaxes = term_freq[term_freq["count"] == 1]
print(f"{len(hapaxes):,} hapaxes ({len(hapaxes) / len(dtm.T) * 100:0.2f}%)")
```

If we plot token counts, a familiar pattern appears: the distribution is
Zipfian.

#```{code-cell}
```
fig, ax = plt.subplots(figsize = (15, 5))
g = sns.lineplot(data = term_freq, x = term_freq.index, y = "count", ax = ax)
g.set(title = "Term counts", xlabel = "Terms", ylabel = "Count", xticks = []);
```

### Weighting with tf-idf


## Vector Space

### The basics

```{code-cell}
columns = ["page", "screen"]
index = ["d1", "d2"]

ps1 = pd.DataFrame([[14, 7], [4, 12]], columns = columns, index = index)
ps1
```

```{code-cell}
:tags: [remove_cell]
def quiver_plot_2d(dtm, d1, d2, axes_max = 15, figsize = (5, 5)):
    """Create a quiver plot from a two-dimensional document-term matrix.

    Parameters
    ----------
    dtm : pd.DataFrame
        The document-term matrix
    d1 : str
        Dimension 1
    d2 : str
        Dimension 2
    axes_max: int
        Maximum extent of the axes
    figsize : tuple
        Figure size
    """
    # Define an origin point
    origin = [[0, 0], [0, 0]]

    # Create a figure and plot values in each dimension with respect to the
    # origin
    fig, ax = plt.subplots(figsize = figsize)
    ax.quiver(*origin, dtm[d1], dtm[d2], scale = 1, units = "xy")
    ax.set(
        xlim = (0, axes_max), ylim = (0, axes_max),
        xticks = range(0, axes_max + 1), yticks = range(0, axes_max + 1),
        xlabel = d1, ylabel = d2
    )

    # Label the quivers
    for doc in dtm.index:
        ax.text(
            dtm.loc[doc, d1], dtm.loc[doc, d2] + 0.5, doc,
            va = "top", ha = "center"
        )

    plt.show()
```

````{dropdown} Show quiver plot code
```py
def quiver_plot_2d(dtm, d1, d2, axes_max = 14, figsize = (5, 5)):
    """Create a quiver plot from a two-dimensional document-term matrix.

    Parameters
    ----------
    dtm : pd.DataFrame
        The document-term matrix
    d1 : str
        Dimension 1
    d2 : str
        Dimension 2
    axes_max: int
        Maximum extent of the axes
    figsize : tuple
        Figure size
    """
    # Define an origin point
    origin = [[0, 0], [0, 0]]

    # Create a figure and plot values in each dimension with respect to the
    # origin
    fig, ax = plt.subplots(figsize = figsize)
    ax.quiver(*origin, dtm[d1], dtm[d2], scale = 1, units = "xy")
    ax.set(
        xlim = (0, axes_max), ylim = (0, axes_max),
        xticks = range(0, axes_max + 1), yticks = range(0, axes_max + 1),
        xlabel = d1, ylabel = d2
    )

    # Label the quivers
    for doc in dtm.index:
        ax.text(
            dtm.loc[doc, d1], dtm.loc[doc, d2] + 0.5, doc,
            va = "top", ha = "center"
        )

    plt.show()
```
````

```{code-cell}
quiver_plot_2d(ps1, "page", "screen")
```

```{code-cell}
ps2 = pd.DataFrame([[8, 14], [4, 12]], columns = columns, index = index)
ps2
```

```{code-cell}
quiver_plot_2d(ps2, "page", "screen")
```

### Vector operations

```{code-cell}
A = np.array([2, 4, 6])
B = np.array([3, 5, 7])
```

**Summation**
+ Description: Element-wise sums
+ Notation: $A + B = (a_1 + b_1, a_2 + b_2, \dots, a_n + b_n)$
+ Result: Vector of length $n$

```{code-cell}
A + B
```

**Subtraction**
+ Description: Element-wise differences
+ Notation: $A - B = (a_1 - b_1, a_2 - b_2, \dots, a_n - b_n)$
+ Result: Vector of length $n$

```{code-cell}
A - B
```

**Multiplication, element-wise**
+ Description: Element-wise products
+ Notation: $A \circ B = (a_1 \cdot b_1, a_2 \cdot b_2, \dots, a_n \cdot b_n)$
+ Result: Vector of length $n$

```{code-cell}
A * B
```

**Multiplication, dot product**
+ Description: The sum of the products
+ Notation: $A \cdot B = \Sigma_{i=1}^{n} a_i \cdot b_i$
+ Result: Single value (scalar)
+ NumPy: `A @ B`

```{code-cell}
A @ B
```

## Cosine Similarity



## Clustering with Cosine Similarity
