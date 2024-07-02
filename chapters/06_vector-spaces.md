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


Vector Space Semantics
======================

## Preliminaries

We need the following libraries:

```{code-cell}
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
```

Our documents have already been vectorized into a document-term matrix (DTM).
The only consideration is that the tokens have been **lemmatized**, that is,
reduced to their uninflected forms.

```{code-cell}
dtm = pd.read_parquet("data/nyt_obituaries/nyt_obituaries_dtm.parquet")
dtm.info()
```


## Vector Space

In the last chapter we plotted Henry James chapters in a two-dimensional
scatter plot. We can do the same with our DTM using the function below.

```{code-cell}
def scatter_2d(matrix, norm = True, highlight = None, figsize = (5, 5)):
    """Plot a matrix in 2D vector space.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to plot
    norm : bool
        Whether to normalize the matrix values
    highlight : None or list
        List of indices to highlight
    figsize : tuple
        Figure size
    """
    # Find a way to reduce the matrix to two dimensions with PCA, adding
    # optional normalization along the way
    parts = (Normalizer(), PCA(2)) if norm else (PCA(2), )
    pipeline = make_pipeline(*parts)
    reduced = pipeline.fit_transform(matrix)
    vis_data = pd.DataFrame(reduced, columns = ["x", "y"])

    # Plot
    fig, ax = plt.subplots(figsize = figsize)
    g = sns.scatterplot(
        x = "x", y = "y", alpha = 0.8, data = vis_data, ax = ax
    )

    # Highlight markers, if applicable
    if highlight:
        selected = vis_data.iloc[highlight]
        sns.scatterplot(
            x = "x", y = "y", color = "red", data = selected, ax = ax
        )

    g.set(xlabel = "Dim. 1", ylabel = "Dim. 2")
    plt.show()
```

Below, we plot the DTM.

```{code-cell}
scatter_2d(dtm)
```

Doing so projects our matrix into a two-dimensional **vector space**. In the
reigning metaphor of NLP, a space of this sort is a stand-in for meaning: the
closer two points are in the scatter plot, the more similar they are in
meaning. 

Selecting the names of two related people in the obituaries will make this
clear.

```{code-cell}
names = ["Bela Bartok", "Maurice Ravel"]
highlight = [dtm.index.get_loc(name) for name in names]
scatter_2d(dtm, highlight = highlight)
```

Let's add in a third that we'd expect to be less similar.

```{code-cell}
names = ["Bela Bartok", "Maurice Ravel", "FDR"]
highlight = [dtm.index.get_loc(name) for name in names]
scatter_2d(dtm, highlight = highlight)
```

While we can only visualize these similarities in two- or three-dimensional
spaces, which are called **Euclidean spaces** (i.e., they conform to physical
space), the same idea---and importantly, the math---holds for similarity in
high-dimensional spaces. But before we turn to what similarity means in vector
space, we'll overview how vectors work generally.

Our two example vectors will be the following:

```{code-cell}
A, B = dtm.loc["Lucille Ball"], dtm.loc["Carl Sagan"]
```

And we will limit ourselves to only two dimensions:

```{code-cell}
terms = ["television", "star"]
A, B = A[terms], B[terms]
```


### Vector components

A vector has a **magnitude** and a **direction**.

**Magnitude**

+ Description: The length of a vector from its origin to its end point. This is
  calculated as the square root of the sum of squares of its components
+ Notation: $||A|| = \sqrt{a_1^2 + a_2^2 + \dots + a_n^2}$
+ Result: Single value (scalar)

```{code-cell}
manual = np.sqrt(np.sum(np.square(A)))
numpy = np.linalg.norm(A)

assert manual == numpy, "Magnitudes don't match!"
print(numpy)
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

Let's plot our two vectors to show their magnitude and orientation.

```{code-cell}
:tags: [remove_cell]
def plot_vectors(
    *vectors,
    vector_labels = [],
    axis_labels = [],
    colors = [],
    figsize = (3, 3)
):
    """Plot 2-dimensional vectors.

    Parameters
    ----------
    vectors : nd.ndarray
        Vectors to plot
    vector_labels : list
        Labels for the vectors
    axis_labels : list
        Labels for the axes in (x, y) order
    colors : list
        Vector colors (string names like "black", "red", etc.)
    figsize : tuple
        Figure size
    """
    # Wrap vectors into a single array
    vectors = np.array(vectors)
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
            label = vector_labels[idx] if vector_labels else None
        )

    # Set plot limits
    limit = np.max(np.abs(vectors))
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])

    # Set axes to be in the center of the plot
    ax.axhline(y = 0, color = "k", linewidth = 0.8)
    ax.axvline(x = 0, color = "k", linewidth = 0.8)

    # Remove the outer box
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add axis labels, if applicable
    if axis_labels:
        xlab, ylab = axis_labels
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    # Add a label legend, if applicable
    if vector_labels:
        ax.legend(loc = "upper left", bbox_to_anchor = (1, 1))

    # Show the plot
    plt.show()
```

````{dropdown} Show vector plot code
```py
def plot_vectors(
    *vectors,
    vector_labels = [],
    axis_labels = [],
    colors = [],
    figsize = (3, 3)
):
    """Plot 2-dimensional vectors.

    Parameters
    ----------
    vectors : nd.ndarray
        Vectors to plot
    vector_labels : list
        Labels for the vectors
    axis_labels : list
        Labels for the axes in (x, y) order
    colors : list
        Vector colors (string names like "black", "red", etc.)
    figsize : tuple
        Figure size
    """
    # Wrap vectors into a single array
    vectors = np.array(vectors)
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
            label = vector_labels[idx] if vector_labels else None
        )

    # Set plot limits
    limit = np.max(np.abs(vectors))
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])

    # Set axes to be in the center of the plot
    ax.axhline(y = 0, color = "k", linewidth = 0.8)
    ax.axvline(x = 0, color = "k", linewidth = 0.8)

    # Remove the outer box
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add axis labels, if applicable
    if axis_labels:
        xlab, ylab = axis_labels
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

    # Add a label legend, if applicable
    if vector_labels:
        ax.legend(loc = "upper left", bbox_to_anchor = (1, 1))

    # Show the plot
    plt.show()
```
````

```{code-cell}
vector_labels = ["L. Ball", "C. Sagan"]
colors = ["red", "blue"]
plot_vectors(
    A, B, vector_labels = vector_labels, axis_labels = terms, colors = colors
)
```

We can normalize our vectors by their direction. This will make the magnitude
of each vector equal to 1. You'll see this operation called **L2
normalization**. (In `scatter_2d` above, this is what the `Normalizer` object
does.)

```{code-cell}
def l2_norm(vector):
    """Perform L2 normalization.

    Parameters
    ----------
    vector : np.ndarray
        Vector to normalize

    Returns
    -------
    vector : np.ndarray
        Normed vector
    """
    norm_by = np.linalg.norm(vector)
    return vector / norm_by
```

```{code-cell}
plot_vectors(
    l2_norm(A),
    l2_norm(B),
    vector_labels = vector_labels, 
    axis_labels = terms,
    colors = colors
)
```


### Vector operations

We turn now to basic operations you can perform on/with vectors.

**Summation**

+ Description: Element-wise sums
+ Notation: $A + B = (a_1 + b_1, a_2 + b_2, \dots, a_n + b_n)$
+ Result: Vector of length $n$

```{code-cell}
C = A + B
plot_vectors(
    l2_norm(A),
    l2_norm(B),
    l2_norm(C),
    vector_labels = vector_labels + ["Result"], 
    axis_labels = terms,
    colors = colors + ["green"]
)
```

**Subtraction**

+ Description: Element-wise differences
+ Notation: $A - B = (a_1 - b_1, a_2 - b_2, \dots, a_n - b_n)$
+ Result: Vector of length $n$

```{code-cell}
C = A - B
plot_vectors(
    l2_norm(A),
    l2_norm(B),
    l2_norm(C),
    vector_labels = vector_labels + ["Result"], 
    axis_labels = terms,
    colors = colors + ["green"]
)
```

**Multiplication, element-wise**

+ Description: Element-wise products
+ Notation: $A \circ B = (a_1 \cdot b_1, a_2 \cdot b_2, \dots, a_n \cdot b_n)$
+ Result: Vector of length $n$

```{code-cell}
C = A * B
plot_vectors(
    l2_norm(A),
    l2_norm(B),
    l2_norm(C),
    vector_labels = vector_labels + ["Result"], 
    axis_labels = terms,
    colors = colors + ["green"]
)
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


## Cosine Similarity

Recall that, initially, we wanted to understand how similar two vectors are. We
can do so with the dot product. The dot product allows us to derive a measure
of how similar two vectors are in their orientation in vector space by
considering the angle formed between them. This measure is called **cosine
similarity**, and it is a quintessential method for working in semantic space.

We express it as follows:

$$
cos \theta = \frac{A \cdot B}{||A||||B||}
$$

Where:

+ $cos \theta$: cosine of the angle $\theta$
+ $A \cdot B$: dot product of A and B
+ $||A||||B||$: the product of A and B's magnitudes

The denominator is a normalization operation, similar in nature to L2
normalization. It helps control for magnitude variance between vectors, which
is important when documents are very different in length.

The function below derives a cosine similarity score for two vectors.

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

    return np.round(cos_sim, 8)
```

Scores are between $[-1, 1]$, where:

+ $1$: same orientation; perfect similarity
+ $0$: orthogonal vectors; vectors have nothing in common
+ $-1$: opposite orientation; vectors are the opposite of one another

Here's A and A:

```{code-cell}
calculate_cosine_similarity(A, A)
```

A and B:

```{code-cell}
calculate_cosine_similarity(A, B)
```

And A and its opposite:

```{code-cell}
calculate_cosine_similarity(A, -A)
```


### Document similarity

Typically, however, you'd just use the `scikit-learn` implementation. It
returns a square matrix comparing each vector with every other vector in the
input. Below, we run it across our DTM.

```{code-cell}
doc_sim = cosine_similarity(dtm)
doc_sim = pd.DataFrame(doc_sim, columns = dtm.index, index = dtm.index)
doc_sim.head()
```

Let's look at some examples. Below, we query for a document's **nearest
neighbors**. That is, we look for which documents are closest to that document
in the vector space.

```{code-cell}
:tags: [output_scroll]
for name in ("Miles Davis", "Eleanor Roosevelt", "Willa Cather"):
    print(doc_sim[name].nlargest(10), end = "\n\n")
```

Does this conform to what we see in the scatter plot of documents?

```{code-cell}
names = doc_sim["Miles Davis"].nlargest(10).index
highlight = [dtm.index.get_loc(name) for name in names]
scatter_2d(dtm, highlight = highlight)
```


### Token similarity

All of the above applies to tokens as well. Transpose the DTM and you can
derive cosine similarity scores between tokens.

```{code-cell}
token_sim = cosine_similarity(dtm.T)
token_sim = pd.DataFrame(token_sim, columns = dtm.columns, index = dtm.columns)
token_sim.sample(5)
```

Similar token listings:

```{code-cell}
:tags: [output_scroll]
for token in ("music", "politics", "country", "royal"):
    print(token_sim[token].nlargest(10), end = "\n\n")
```

We can also project these vectors into a two-dimensional space. The code below
extracts the top-10 nearest tokens for a query token, samples the cosine
similarities, adds the two together into a single DataFrame, and plots them.
Note that we turn off normalization; cosine similarity is already normed.

```{code-cell}
# Get the subset, then sample
subset = token_sim["royal"].nlargest(10).index
sample = token_sim.sample(500)
sample = sample[~sample.index.isin(subset)]

# Make the DataFrame for plotting
to_plot = pd.concat([token_sim.loc[subset], sample])
highlight = [to_plot.index.get_loc(name) for name in subset]

# Plot
scatter_2d(to_plot, norm = False, highlight = highlight)
```

Technically, one could call these vectors definitions of each token. But they
aren't very good definitions---for a number of reasons, the first of which is
that the only data we have to create these vectors comes from a small corpus of
obituaries. We'd need a much larger corpus to generalize our token
representations to a point where these vectors might reflect semantics as we
understand it.


## Word Embeddings

```{code-cell}
class WordEmbeddings:
    """A minimal wrapper for Word2Vec-style embeddings.
    
    Loosely based on the Gensim wrapper: https://radimrehurek.com/gensim.
    """
    def __init__(self, path):
        """Initialize the embeddings.

        Parameters
        ----------
        path : str or Path
            Path to the embeddings parquet file
        """
        # Load the data and store some metadata about the vocabulary
        self.embeddings = pd.read_parquet(path)
        self.vocab = self.embeddings.index.tolist()
        self.vocab_size = len(self.vocab)

        # Fit a nearest neighbors graph using cosine similarity
        self.neighbors = NearestNeighbors(
            metric = "cosine", n_neighbors = self.vocab_size, n_jobs = -1
        )
        self.neighbors.fit(self.embeddings)

    def __getitem__(self, key):
        """Index the embeddings to retrieve a vector.

        Parameters
        ----------
        key : str
            Index key

        Returns
        -------
        vector : np.ndarray
            The vector
        """
        if key not in self.vocab:
            raise KeyError(f"{key} not in vocabulary.")

        return np.array(self.embeddings.loc[key])

    def most_similar(self, query, k = 1):
        """Find the k-most similar tokens to a query vector.

        Parameters
        ----------
        query : str or np.ndarray
            A query string or vector
        k : int
            Number of neighbors to return

        Returns
        -------
        output : list[tuple]
            Nearest tokens and their similarity scores
        """
        # If passed a string, get the vector. Reshape this vector for the
        # nearest neighbor graph
        if isinstance(query, str):
            query = self[query]
        vector = np.array(query).reshape(1, -1)

        # Query the nearest neighbor graph. This returns the index positions
        # for the top-k most similar tokens and the distance values for each of
        # those tokens 
        distances, knn = self.neighbors.kneighbors(vector, n_neighbors = k)

        # Convert distances to similarities and squeeze out the extra
        # dimension. Then retrieve tokens (and squeeze out that dimension, too)
        similarities = (1 - distances).squeeze(0)
        tokens = self.embeddings.iloc[knn.squeeze(0)].index

        # Pair the tokens with the similarities
        output = [(tok, sim) for tok, sim in zip(tokens, similarities)]

        return output

    def analogize(self, this = "", to_that = "", as_this = "", k = 1):
        """Perform an analogy and return the result's k-most similar neighbors.

        Parameters
        ----------
        this : str
            First term of analogy's source
        to_that : str
            Second term of analogy's source
        as_this : str
            First term of analogy's target
        k : int
            Number of neighbors to return

        Returns
        -------
        output : list[tuple]
            Nearest tokens and their similarity scores
        """
        # Get the vectors for input terms
        this, to_that, as_this = self[this], self[to_that], self[as_this]

        # Subtract the first term of the analogy's source from the analogy's
        # target term to capture the relationship between the two (that is,
        # their difference). Then, apply this relationship to the second term
        # of the analogy's source by adding it
        is_to_what = (as_this - this) + to_that

        # Get the most similar tokens to this new vector
        return self.most_similar(is_to_what, k)
```

```{code-cell}
glove = WordEmbeddings("data/glove/glove.6B.200d.parquet")
_, n_dim = glove.embeddings.shape
print(f"Embeddings shape: {n_dim}")
```


## Part-of-Speech Classification


### Data preparation

<!-- Data: Embeddings for POS tagged WordNet lemmas --->

```{code-cell}
wordnet = pd.read_parquet("data/glove/wordnet_embeddings.parquet")
```

Words can have more than one POS tag.

```{code-cell}
wordnet.loc[("good", slice(None))]
```

But the vectors are the same because static embeddings make no distinction in
context-dependent determinations. We therefore drop duplicate vectors from our
data. First though: we shuffle the rows. That will randomly distribute the POS
tag distribution across all duplicated vectors (in other words, no one POS tag
will unfairly benefit from dropping duplicates).

```{code-cell}
wordnet = wordnet.sample(frac = 1)
wordnet.drop_duplicates(inplace = True)
```

Now split the data into train/test sets. Our labels are the POS tags.

```{code-cell}
X = wordnet.values
y = wordnet.index.get_level_values(1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.1, random_state = 357
)
```

Regardless of duplication, the tags themselves are fairly imbalanced.

```{code-cell}
tag_counts = pd.Series(y).value_counts()
tag_counts
```

We derive a weighting to tell the model how often it should expect to see a
tag.

```{code-cell}
class_weight = tag_counts / np.sum(tag_counts)
class_weight = {
    idx: np.round(class_weight[idx], 2) for idx in class_weight.index
}
class_weight
```


### Model training

Now initialize the model and fit it.

```{code-cell}
model = LogisticRegression(
    solver = "liblinear",
    C = 2.0,
    class_weight = class_weight,
    max_iter = 5_000,
    random_state = 357
)
model.fit(X_train, y_train)
```

Let's look at a classification report.

```{code-cell}
preds = model.predict(X_test)
report = classification_report(
    y_test, preds, target_names = ["adj", "noun", "verb"]
)
print(report)
```

Not bad. For a such a simple model, it achieves decent accuracy, and the
precision is relatively high. Recall suffers, however. This is likely due to
the class imbalance.


### Examining model coefficients

Let's extract the coefficients from the model. These represent the weightings
on each dimension of the word vectors that adjust the associated values to be
what the model expects for a certain class.

+ **Positive coefficient:** An increase in the feature value increases the
  log-odds of the classification outcome; the probability of a token belonging
  to the class increases
+ **Negative coefficient:** An increase in the feature value decreases the
  log-odds of the classification outcome; the probability of a token belonging
  to the class decreases

```{code-cell}
coef = pd.DataFrame(model.coef_.T, columns = ["adj", "noun", "verb"])
coef.head()
```

We'll use this DataFrame in a moment, but for now let's reformat to create a
series of bar plots showing how each dimension of the embeddings interacts with
the model's coefficients.

```{code-cell}
coef_plot = coef.reset_index().rename(columns = {"index": "dimension"})
coef_plot = pd.melt(
    coef_plot,
    id_vars = "dimension",
    var_name = "POS",
    value_name = "coefficient"
)
```

Time to plot.

```{code-cell}
fig, axes = plt.subplots(3, 1, figsize = (15, 5), sharex = True, sharey = True)
for idx, pos in enumerate(["adj", "noun", "verb"]):
    subplot_data = coef_plot[coef_plot["POS"] == pos]
    g = sns.barplot(
        x = "dimension",
        y = "coefficient",
        data = coef_plot[coef_plot["POS"] == pos],
        ax = axes[idx]
    )
    g.set(ylabel = f"{pos} coefficient", xticklabels = [])

plt.suptitle("POS coefficients across word embedding dimensions")
plt.tight_layout()
plt.show()
```

No one single dimension stands out as the indicator for a POS tag; rather,
groups of dimensions determine POS tags. This is in part what we mean by a
distributed representation. We can use a `SelectFromModel` object to extract
dimensions that are particularly important. Use `threshold` to set a cutoff
value below which features are considered unimportant (the threshold assumes
absolute values).

```{code-cell}
selector = SelectFromModel(
    model, threshold = 1e-5, prefit = True, max_features = 10
)
selector.fit(wordnet)
```

Collect the dimension names from the feature selector and plot.

```{code-cell}
important_features = selector.get_support(indices = True)
fig, ax = plt.subplots(figsize = (15, 5))
g = sns.barplot(
    x = "dimension",
    y = "coefficient",
    hue = "POS",
    palette = "tab10",
    data = coef_plot[coef_plot["dimension"].isin(important_features)],
    ax = ax
    
)
g.set(title = "Ten most important features for determining a POS tag")
sns.move_legend(ax, "upper left", bbox_to_anchor = (1, 1))
plt.show()
```


### Tag associations

If we multiply the vectors by the model coefficients, we can determine the
degree to which each sample is associated with a POS tag. This will concretely
ground the above graphs in specific tokens.

```{code-cell}
associations = np.dot(wordnet, coef)
associations = pd.DataFrame(
    associations, index = wordnet.index, columns = ["adj", "noun", "verb"]
)
associations.sample(5)
```

Let's look at some tokens. Below, we show tokens that are unambiguously one
kind of word or another. Using the `.idxmax()` method will return which column
has the highest coefficient.

```{code-cell}
inspect = ["colorless", "lion", "recline"]
associations.loc[(inspect, slice(None))].idxmax(axis = 1)
```

That seems to work pretty well! But let's look at a word that can be more than
one POS.

```{code-cell}
inspect = ["good", "goad", "great"]
associations.loc[(inspect, slice(None))].idxmax(axis = 1)
```

Here we see the constraints of static embeddings. The coefficients for these
words favor one POS over another, and that should make sense: the underlying
embeddings make no distinction between two. One reason why our classifier
doesn't always perform well very likely has to do with this as well.

That said, getting class probabilities tells a slightly more encouraging story. 

```{code-cell}
for token in inspect:
    # Reshape and squeeze when making single vector predictions
    vector = glove[token].reshape(1, -1)
    probs = model.predict_proba(vector).squeeze(0)

    print(f"Predictions for '{token}':")
    for idx, label in enumerate(["adj", "noun", "verb"]):
        print(f"  {label}: {probs[idx]:.2f}%")
```

The label with the second-highest probability tends to be for a token's other
POS tag.


### Shifting vectors

The above probabilities suggest that a token is somewhat like one POS and
somewhat like another. On the one hand that conforms with how we understand POS
tags to work; but on the other, it matches the additive nature of logistic
regression: each set of coefficients indicates how much a given feature in the
vector contributes to a particular class prediction. This property about
coefficients also suggests a way to modify vectors so that they are more like
one POS tag or another.

The function below does just that. It shifts the direction of a vector in
vector space so that its orientation is more like a particular POS tag. How
does it do so? With simple addition: we add the coefficients for our desired
POS tag to the vector in question.

```{code-cell}
def shift_vector(vector, coef, glove = glove, k = 25):
    """Shift the direction of a vector by combining it with a coefficient.

    Parameters
    ----------
    vector : nd.ndarray
        The vector to shift
    coef : np.ndarray
        Coefficient vector
    glove : WordEmbeddings
        The word embeddings
    k : int
        Number of nearest neighbors to return

    Returns
    -------
    output : pd.DataFrame
        A DataFrame containing k-nearest neighbors for the original and shifted
        vectors
    """
    # Find the vector's k-nearest neighbors
    vector_knn = glove.most_similar(vector, k)

    # Now shift it by adding the coefficient. Then find the k-nearest neighbors
    # of this new vector
    shifted = vector + coef
    shifted_knn = glove.most_similar(shifted, k)

    # Extract the tokens and put them into a DataFrame
    output = [
        (tok1, tok2) for (tok1, _), (tok2, _) in zip(vector_knn, shifted_knn)
    ]
    output = pd.DataFrame(output, columns = ["original", "shifted"])
    output.index.name = "k-th_neighbor"

    return output
```

Let's try this with a few tokens. Below, we make "dessert" more like an
adjective.

```{code-cell}
:tags: [output_scroll]
shift_vector(glove["dessert"], coef["adj"])
```

How about "desert"?

```{code-cell}
:tags: [output_scroll]
shift_vector(glove["desert"], coef["adj"])
```

Now make "language" more like a verb.

```{code-cell}
:tags: [output_scroll]
shift_vector(glove["language"], coef["verb"])
```

The modifications here are subtle, but they do exist: while the top-most
similar tokens tend to be the same between the original and shifted vectors,
their ordering moves around, often in ways that conform to what you'd expect to
see with the POS's valences.

```{code-cell}
:tags: [output_scroll]
shift_vector(glove["drive"], coef["noun"])
```
