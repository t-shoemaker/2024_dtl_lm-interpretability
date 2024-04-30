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
data = pd.read_parquet("data/nyt_obituaries/nyt_obituaries.parquet")
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

### Weighting with tf-idf


## Vector Space

### The basics

```{code-cell}
columns = ["page", "screen"]
index = ["doc1", "doc2"]

ps1 = pd.DataFrame([[4, 1], [2, 3]], columns = columns, index = index)
ps1
```

```{code-cell}
:tags: [remove_cell]
def plot_vectors(vectors, labels = [], colors = [], figsize = (4, 4)):
    """Plot 2-dimensional vectors.

    Parameters
    ----------
    vectors : np.ndarray
        Vectors to plot
    labels : list
        Labels for the vectors
    colors : list
        Vector colors (string names like "black", "red", etc)
    figsize : tuple
        Figure size
    """
    n_vector, n_dim = vectors.shape
    if n_dim != 2:
        raise ValueError("We can only plot 2-dimensional vectors")

    # Populate colors
    if not colors:
        colors = ["black"] * n_vector

    # Create a (0, 0) origin point for each vector
    origin = np.zeros((2, n_vector))

    # Then plot each vector
    fig, ax = plt.subplots(figsize = figsize)
    for idx, vector in enumerate(vectors):
        color = colors[idx]
        ax.quiver(
            *origin[:, idx],
            vector[0],
            vector[1],
            color = color,
            scale = 1,
            units = "xy",
            label = labels[idx] if labels else None
        )

    # Set plot limits
    limit = np.max(np.abs(vectors)) + 1
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])

    # Set ticks
    ax.set_xticks(np.arange(-limit, limit + 1, 1))
    ax.set_yticks(np.arange(-limit, limit + 1, 1))
    ax.set_aspect("equal")

    # Set axes to be in the center of the plot
    ax.axhline(y = 0, color = "k", linewidth = 0.8)
    ax.axvline(x = 0, color = "k", linewidth = 0.8)

    # Add a label legend, if applicable
    if labels:
        ax.legend()

    # Show the plot
    plt.show()
```

````{dropdown} Show vector plot code
```py
def plot_vectors(vectors, labels = [], colors = [], figsize = (4, 4)):
    """Plot 2-dimensional vectors.

    Parameters
    ----------
    vectors : np.ndarray
        Vectors to plot
    labels : list
        Labels for the vectors
    colors : list
        Vector colors (string names like "black", "red", etc)
    figsize : tuple
        Figure size
    """
    n_vector, n_dim = vectors.shape
    if n_dim != 2:
        raise ValueError("We can only plot 2-dimensional vectors")

    # Populate colors
    if not colors:
        colors = ["black"] * n_vector

    # Create a (0, 0) origin point for each vector
    origin = np.zeros((2, n_vector))

    # Then plot each vector
    fig, ax = plt.subplots(figsize = figsize)
    for idx, vector in enumerate(vectors):
        color = colors[idx]
        ax.quiver(
            *origin[:, idx],
            vector[0],
            vector[1],
            color = color,
            scale = 1,
            units = "xy",
            label = labels[idx] if labels else None
        )

    # Set plot limits
    limit = np.max(np.abs(vectors)) + 1
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])

    # Set ticks
    ax.set_xticks(np.arange(-limit, limit + 1, 1))
    ax.set_yticks(np.arange(-limit, limit + 1, 1))
    ax.set_aspect("equal")

    # Set axes to be in the center of the plot
    ax.axhline(y = 0, color = "k", linewidth = 0.8)
    ax.axvline(x = 0, color = "k", linewidth = 0.8)

    # Add a label legend, if applicable
    if labels:
        ax.legend()

    # Show the plot
    plt.show()
```
````

```{code-cell}
plot_vectors(ps1.values, labels = list(ps1.index), colors = ["red", "blue"])
```

```{code-cell}
ps2 = pd.DataFrame([[1, 4], [2, 3]], columns = columns, index = index)
ps2
```

```{code-cell}
plot_vectors(ps2.values, labels = list(ps2.index), colors = ["red", "blue"])
```

### Vector operations

```{code-cell}
A = ps1.loc["doc1"].values
B = ps1.loc["doc2"].values
```

A vector has a **magnitude** and a **direction**.

**Magnitude**
+ Description: The length of a vector from its origin to its end point. This is
  calculated as the square root of the sum of squares of its components
+ Notation: $||A|| = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2}$
+ Result: Single value (scalar)

```{code-cell}
manual = np.sqrt(np.sum(np.square(A)))
shorthand = np.linalg.norm(A)

assert manual == shorthand, "Magnitudes don't match!"
print(shorthand)
```

**Direction**
+ Description: The orientation of a vector in space. In Cartesian coordinates,
  it is the angles a vector forms across its axes. But direction can also be
  represented as a second **unit vector**, a vector of magnitude 1 that points
  in the same direction as the first. Most vector operations in NLP use the
  latter
+ Notation: $\hat{A} = \frac{A}{||A||}$
+ Result: Vector of length $n$

```{code-cell}
A / np.linalg.norm(A)
```

**Summation**
+ Description: Element-wise sums
+ Notation: $A + B = (a_1 + b_1, a_2 + b_2, \dots, a_n + b_n)$
+ Result: Vector of length $n$

```{code-cell}
C = A + B
plot_vectors(np.array([A, B, C]), colors = ["red", "blue", "green"])
```

**Subtraction**
+ Description: Element-wise differences
+ Notation: $A - B = (a_1 - b_1, a_2 - b_2, \dots, a_n - b_n)$
+ Result: Vector of length $n$

```{code-cell}
C = A - B
plot_vectors(np.array([A, B, C]), colors = ["red", "blue", "green"])
```

**Multiplication, element-wise**
+ Description: Element-wise products
+ Notation: $A \circ B = (a_1 \cdot b_1, a_2 \cdot b_2, \dots, a_n \cdot b_n)$
+ Result: Vector of length $n$

```{code-cell}
C = A * B
plot_vectors(np.array([A, B, C]), colors = ["red", "blue", "green"])
```

**Multiplication, dot product**
+ Description: The sum of the products
+ Notation: $A \cdot B = \Sigma_{i=1}^{n} a_i \cdot b_i$
+ Result: Single value (scalar)

```{code-cell}
A @ B
```

The dot product is one of the most important operations in modern machine
learning. It measures the extent to which two vectors point in the same
direction. If the dot product is positive, the angle between two vectors is
under 90 degrees. This means they point somewhat in the same direction. If it
is negative, they point in opposite directions. And when the dot product is
zero, the vectors are perpendicular.


### Vector comparisons

Dot product enables a variety of comparisons across vectors.

### Projection

```{code-cell}
def project_vector(A, B):
    """Project vector A onto vector B.

    Parameters
    ----------
    A : np.ndarray
        Vector to project
    B : np.ndarray
        Vector on which to project A

    Returns
    -------
    projection : np.ndarray
        The component of A that lies in the direction of B
    """
    # Compute the dot product of the two vectors
    dot = A @ B
    
    # Normalization: the magnitude (length) of squared B. Applying normalization
    # strips out difference in magnitude between two vectors so their directions
    # are comparable
    normalization = B @ B
    
    # Apply normalization. This adjusts the length of B to match that part of A
    # that lies in the same direction
    scale_factor = dot / normalization

    # Scale B
    projection = scale_factor * B

    return projection
```

### Cosine Similarity

```{code-cell}
def calculate_cosine_similarity(A, B):
    """Calculate cosine similarity between two vectors.

    Parameters
    ----------
    A : np.ndarray
        First vector
    B : np.ndarray
        Second vector

    Returns
    -------
    cos_sim : float
        Cosine similarity of A and B
    """
    dot = A @ B
    norm_by = np.linalg.norm(A) * np.linalg.norm(B)
    cos_sim = dot / norm_by

    return cos_sim
```

## Clustering with Cosine Similarity
