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

This chapter introduces vectorization, a technique for encoding qualitative
data (like words) into numeric values. We will use a data structure, the
document-term matrix, to work with vectorized texts, discuss weighting
strategies for managing high-frequency tokens, and train a classification model
to distinguish style.

+ **Data**: 20 Henry James novels, collected by [Jonathan Reeve][reeve] and
  broken into chapters with Reeve's [chapterization][chapterize] tool. Labels
  are from David L. Hoover's [clustering][clustering] of James's novels

[reeve]: https://github.com/JonathanReeve/james-sentence
[chapterize]: https://github.com/JonathanReeve/chapterize
[clustering]: https://dlsanthology.mla.hcommons.org/textual-analysis/


## Preliminaries

We will need the following libraries:

```{code-cell}
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
```

Corpus documents are stored in a DataFrame alongside other metadata.

```{code-cell}
corpus = pd.read_parquet("data/datasets/james_chapters.parquet")
print(corpus.info())
```

Novels are divided into their component chapters. Use `.groupby()` to count how
many chapters there are for each novel.

```{code-cell}
grouped = corpus.groupby("novel")
chapters_per_novel = grouped["chapter"].count()
chapters_per_novel.to_frame(name = "chapters")
```

The `style` and `hoover` columns contain labels. The first demarcates early
James from late with the publication of _What Maisie Knew_ in 1897.

```{code-cell}
style_counts = grouped[["year", "style"]].value_counts()
style_counts = style_counts.to_frame(name = "chapters")
style_counts.sort_values("style")
```

The second uses Hoover's grouping of James's novels into four distinct phases.

```{code-cell}
hoover_counts = grouped[["year", "hoover"]].value_counts()
hoover_counts = hoover_counts.to_frame(name = "chapters")
hoover_counts.sort_values("hoover")
```

Tokens for each chapter are stored as strings in `tokens` and `masked`.
Chapters have been tokenized with `nltk.wordpunct_tokenize()`. Why `masked`?
That column has had its proper noun tokens masked out with `PN`. You will see
why later on.

````{dropdown} See masking code
```py
def mask_proper_nouns(string, mask = "PN"):
    """Mask proper nouns in a string.

    Parameters
    ----------
    string : str
        String to mask
    mask : str
        Masking label

    Returns
    -------
    masked : str
        The masked string
    """
    # First, split the string into tokens
    tokens = string.split()

    # Then assign part-of-speech tags to those tokens. The output of this
    # tagger is a list of tuples, where the first element is the token and the
    # second is the tag
    tagged = nltk.pos_tag(tokens)

    # Create a new list to hold the output
    masked = []
    for (token, tag) in tagged:
        # If the tag is "NNP", replace it with our mask
        token = mask if tag == "NNP" else token
        # Add the token to the output list
        masked.append(token)

    # Join the list and return
    masked = " ".join(masked)

    return masked
    
```
````


## The Document-Term Matrix

So far we have worked with lists of tokens. That works for some tasks, but to
compare documents with one another, it would be better to represent our corpus
as a two-dimensional array, or matrix. In this matrix, each row is a document
and each column is a token; cells record the number of times that token appears
in a document. The resultant matrix is known as the **document-term matrix**,
or DTM. 

It isn't difficult to convert lists of tokens into a DTM, but `scikit-learn`
can do it. Unless you have a reason to convert your token lists manually, just
rely on that.

````{dropdown} See function to create a document-term matrix by hand
```py
def make_dtm(docs):
    """Make a document-term matrix.

    Parameters
    ----------
    docs : list[str]
        A list of strings

    Returns
    -------
    dtm, vocabulary : tuple[list, set]
        The document-term matrix and the corpus vocabulary
    """
    # Split the documents into tokens
    docs = [doc.split() for doc in docs]

    # Get the unique set of tokens for all documents in the corpus
    vocab = set()
    for doc in docs:
        # A set union (`|=`) adds any new tokens from the current document to
        # the running set of all tokens
        vocab |= set(doc)

    # Create a list of m dictionaries, where m is the number of corpus
    # documents. Each dictionary will have every token in the vocabulary (key),
    # which is initially assigned to 0 (value)
    counts = [dict.fromkeys(vocab, 0) for doc in docs]

    # Roll through each document
    for idx, doc in enumerate(docs):
        # For each token in a document...
        for tok in doc:
            # Access the document counts, then access the token stored in the
            # dictionary. Increment the corresponding count by 1 
            counts[idx][tok] += 1

    # Extract the values from each dictionary
    dtm = [[count for count in doc.values()] for doc in counts]

    # Return the DTM and the vocabulary
    return dtm, vocab
```
````

Many classes in `scikit-learn` have the same use pattern: first, you initialize
the class by assigning it to a variable (and optionally set parameters), then
you **fit** it on your data. The `CountVectorizer`, which makes a DTM, does
just this. It will even tokenize strings while it fits, though watch out: it
has a simple tokenization pattern, so it's often best to do this step yourself.

Below, we initialize the `CountVectorizer` and set the following parameters:

+ `token_pattern`: a regex pattern for which tokens to keep (here: any
   alphabetic characters of three or more characters)
+ `stop_words`: remove function words in English
+ `strip_accents`: normalize accents to ASCII

```{code-cell}
cv_parameters = {
    "token_pattern": r"\b[a-zA-Z]{3,}\b",
    "stop_words": "english",
    "strip_accents": "ascii"
}

count_vectorizer = CountVectorizer(**cv_parameters)
count_vectorizer.fit(corpus["tokens"])
```

With the vectorizer fitted, **transform** the data you fitted it on.

```{code-cell}
dtm = count_vectorizer.transform(corpus["tokens"])
```

DTMs are **sparse**. That is, they are mostly made up of zeros.

```{code-cell}
dtm
```

This sparsity is significant. Comparing documents with each other requires
taking into account all unique tokens in the corpus, not just those in a
particular document. _This means we must count the number of times a token
appears in a document even if that count is zero_. What those zero counts also
mean is that the documents in a DTM are not strictly those documents that are
in the corpus. They are potential texts: possible distributions of tokens
across the corpus.

The output of `CountVectorizer` is optimized for keeping the memory footprint
of a DTM low. But for a small corpus like this, use `.toarray()` to convert the
matrix into a NumPy array. 

```{code-cell}
dtm = dtm.toarray()
```

Now, wrap this as a DataFrame and set the column names with the output of the
vectorizer's `.get_feature_names_out()` method.

```{code-cell}
dtm = pd.DataFrame(
    dtm, columns = count_vectorizer.get_feature_names_out()
)
dtm.head()
```

This DTM is indexed in the same order as the corpus documents. But for
readability's sake, set the index to the `novel` and `chapter` columns of our
corpus DataFrame. Be sure to change the `.names` attribute of the index, or
your index will conflict with possible column values in the DTM.

```{code-cell}
dtm.index = pd.MultiIndex.from_arrays(
    [corpus["novel"], corpus["chapter"]],
    names = ["novel_name", "chapter_num"]
)
dtm.head()
```

### Document-term matrix analysis

Numeric operations across the DTM now work the same as they would for any other
DataFrame. Here, total tokens per novel:

```{code-cell}
chapter_token_count = np.sum(dtm, axis = 1).to_frame(name = "token_count")
chapter_token_count.groupby("novel_name").sum()
```

On average, which three chapters are the longest in James's novels?

```{code-cell}
chapter_avg_tokens = chapter_token_count.groupby("chapter_num").mean()
chapter_avg_tokens.sort_values("token_count", ascending = False).head(3)
```

What is the average chapter length?

```{code-cell}
chapter_avg_tokens.mean()
```

Top ten chapters with the highest type counts:

```{code-cell}
num_types = (dtm > 0).sum(axis = 1).to_frame(name = "num_types")
num_types.nlargest(10, "num_types")
```

Most frequent word in each novel:

```{code-cell}
token_freq = dtm.groupby("novel_name").sum()
token_freq.idxmax(axis = 1).to_frame(name = "most_frequent_token")
```

All are proper nouns, which often happens with fiction. In one way, this is
valuable information: if you were modeling a corpus with different kinds of
documents, you might use names' frequency to distinguish fiction. But we only
have James's novels, and the high frequency of names can make it difficult to
identify similarities across documents.


### Using masked tokens

This is where the text stored in `masked` comes in. That text masks over proper
nouns and treats them all like the same token. More, due to the way we've
currently configured our DTM generation, those masks will be dropped because
they're only two characters long. That's perfectly fine for our purposes. But
it again underscores the fact that documents in the DTM are not documents as
they are in the corpus. Indeed, through these preprocessing decisions we have
already constructed a model of our corpus.

Time to rebuild the DTM with text in `masked`.

```{code-cell}
count_vectorizer = CountVectorizer(**cv_parameters)
count_vectorizer.fit(corpus["masked"])
dtm = count_vectorizer.transform(corpus["masked"])
```

Convert to a DataFrame.

```{code-cell}
dtm = pd.DataFrame(
    dtm.toarray(), columns = count_vectorizer.get_feature_names_out()
)
dtm.index = pd.MultiIndex.from_arrays(
    [corpus["novel"], corpus["chapter"]],
    names = ["novel_name", "chapter_num"]
)
```

We won't step through the above metrics again, except we will look at top token
counts to confirm that masking made a difference.

```{code-cell}
token_freq = dtm.groupby("novel_name").sum()
token_freq.idxmax(axis = 1).to_frame(name = "most_frequent_token")
```

Names are gone but the output looks even worse. There's no differentiation
among the most frequent tokens in each novel, even when controlling for common
deictic words with stopword removal. Given the nature of Zipfian distributions,
this shouldn't be surprising.

One way to control for this would be to remove tokens from consideration when
building the DTM using some cutoff metric. That would work okay but it may
remove valuable information from the documents. Consider, for example, the fact
that James's penchant for extended psychological descriptions could be usefully
counterposed with chapters with more dialogue. Removing "said" would make it
difficult to do this. More, setting the cutoff point could take a fair bit of
back and forth. A better strategy would be to **re-weight** token counts by
some method so that frequent tokens have less impact in aggregate analyses like
the above.


## Weighting with TF--IDF

This is where TF--IDF, or **term frequency--inverse document frequency**, comes
in. It re-weights tokens according to their specificity in a document. Tokens
that frequently appear in many documents will have low TF--IDF scores, while
those that are less frequent, or appear frequently in only a few documents,
will have high TF--IDF scores.

Scores are the product of two statistics: **term frequency** and **inverse
document frequency**. There are several variations for calculating both but
generally they work like so:

Term frequency is the relative frequency of a token $t$ in a document $d$.

$$
TF(t, d) = \frac{f_{t,d}}{\sum_{i=1}^nf_{i,d}}
$$

Where:
+ $f_{t,d}$ is the frequency of token $t$ in document $d$
+ $\sum_{i=1}^nf_{i,d}$ is the sum of all token frequencies in document $d$

In code, that looks like the following:

```{code-cell}
TF = dtm.div(dtm.sum(axis = 1), axis = 0)
```

Inverse document frequency measures how common or rare a token is.

$$
IDF(t, D) = log\left({\frac{N}{|\{d \in D : t \in d\}|}} \right)
$$

Where:
+ $N$ is the total number of documents in a corpus $D$
+ For each document $d$ in $D$, we count which ones contain token $t$

The code for this calculation is below. Note that we typically add one to the
document frequency to avoid zero-division errors. Adding one outside the
logarithm ensures that any terms that appear across all documents do not
completely zero-out.

```{code-cell}
N = len(dtm)
DF = (dtm > 0).sum(axis = 0)
IDF = np.log(1 + N / (1 + DF)) + 1
```

The product of these two statistics is TF--IDF.

$$
TFIDF(t, d, D) = TF(t, d, D) \cdot IDF(t, D)
$$

Or, in code:

```{code-cell}
TFIDF = TF.multiply(IDF, axis = 1)
```

Don't want to go through all those steps? Use `TfidfVectorizer`. But note that
`scikit-learn` has set some defaults for smoothing/normalizing TF--IDF scores
that could make the result slightly different than your own calculations.

Fitting `TfidfVectorizer` works with the same use pattern.

```{code-cell}
tfidf_vectorizer = TfidfVectorizer(**cv_parameters)
tfidf_vectorizer.fit(corpus["masked"])
tfidf = tfidf_vectorizer.transform(corpus["masked"])
```

Convert to a DataFrame:

```{code-cell}
tfidf = pd.DataFrame(
    tfidf.toarray(), columns = tfidf_vectorizer.get_feature_names_out()
)
tfidf.index = pd.MultiIndex.from_arrays(
    [corpus["novel"], corpus["chapter"]],
    names = ["novel_name", "chapter_num"]
) 
```

And now, finally, the highest scoring tokens for every novel. Again, these are
the most specific tokens.

```{code-cell}
max_tfidf_per_novel = tfidf.groupby("novel_name").max()
max_tfidf_per_novel.idxmax(axis = 1).to_frame(name = "top_token")
```

Top-scoring tokens for each chapter in _What Maisie Knew_. Use an empty `slice`
to get all entries in the second of the two DataFrame indexes.

```{code-cell}
:tags: [output_scroll]
maisie = tfidf.loc[("What Maisie Knew", slice(None))]
maisie_max = pd.DataFrame({
    "token": maisie.idxmax(axis = 1),
    "tfidf": maisie.max(axis = 1)
})
maisie_max
```


## Document Classification

Each document in the weighted DTM is now a **feature vector**: a sequence of
values that encode information about token distributions. These vectors allow
us to estimate joint probabilities between features, which enables
classification tasks.


### The Multinomial Naive Bayes classifier

We use a **Multinomial Naive Bayes** model to classify documents according to
their assigned label, or class. The model trains by calculating the **prior
probability** for each class. Then, it computes the **posterior probability**
of each token given a class. The class with the highest posterior probability
is selected as the label for a document.

| Term                                   | Definition                                        |
|----------------------------------------|---------------------------------------------------|
| Naive Bayes                            | Assumes conditionally independent features        |
| Multinomial distribution               | Models probabilities of counts across categories  |
| Prior probability                      | Probability of an event before observing new data |
| Posterior probability                  | Probability of an event after observing new data  |
| Argmax (maximum likelihood estimation) | Predicts class with highest posterior probability |

The formula for our classifier is as follows:

$$
P(C_k|x) \propto P(C_k) \prod_{i=1}^n P(x_i|C_k)
$$

Where:

+ $P(C_k)$: prior probability of class $C_k$
+ $P(x_i|C_k)$: posterior probability of feature $x_i$ given class $C_k$
+ $P(C_k|x)$: probability of feature vector $x$ being class $C_k$ 


### Training a classifier

No need to do this math ourselves; `scikit-learn` can do it. But first, we
split our data and their corresponding labels into **training** and **testing**
datasets. The model will train on the former, and we will validate that model
on the latter (which is data it hasn't yet seen).

```{code-cell}
X_train, X_test, y_train, y_test = train_test_split(
    tfidf, corpus["hoover"], test_size = 0.3, random_state = 357
)
print(f"Train set size: {len(X_train)}\nTest set size: {len(X_test)}")
```

Train the model using the same initialization/fitting pattern from before.

```{code-cell}
classifier = MultinomialNB(alpha = 0.005)
classifier.fit(X_train, y_train)
```


### Model diagnostics

Use the `.score()` method to return the mean accuracy for all labels given test
data. This is the number of correct predictions divided by the total number of
true labels.

```{code-cell}
accuracy = classifier.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}%")
```

Generate a classification report to get a class-by-class summary of the
classifier's performance. This requires you to make predictions on the test
size, which you then compare against the true labels.

```{code-cell}
preds = classifier.predict(X_test)
```

Now make and print the report.

```{code-cell}
periods = ["1871-81", "1886-90", "1896-99", "1901-17"]
report = classification_report(y_test, preds, target_names = periods)
print(report)
```

The above scores describe trade-offs between the following kinds of
predictions:

| Prediction     | Explanation                            | Shorthand |
|----------------|----------------------------------------|-----------|
| True positive  | Correctly predicts class of interest   | TP        |
| True negative  | Correctly predicts all other classes   | TN        |
| False positive | Incorrectly predicts class of interest | FP        |
| False negative | Incorrectly predicts all other classes | FN        |

Here is a breakdown of score types:

| Score     | Explanation                            | Formula                                |
|-----------|----------------------------------------|----------------------------------------|
| Precision | Accuracy of positive predictions       | $P = \frac{TP}{TP + FP}$               |
| Recall    | Ability to find all relevant instances | $R = \frac{TP}{TP + FN}$               |
| F1        | A balance of precision and recall      | $F1 = 2\times \frac{P\times R}{P + R}$ |

A **weighted** average of these scores offsets each class by its proportion in
the testing set; the **macro** average reports scores with no weighting. The
**support** for each class is the number of documents labeled with that class.

This model does extremely well. In fact, it may be a touch **overfitted**: too
closely matched with its training data and therefore incapable of generalizing
beyond that data. For our purposes this is less of a concern because the corpus
and analysis are both constrained, but you might be suspicious of high-scoring
results like this in other cases.


### Top tokens per class

Recall that the classifier makes its decisions based on the posterior
probability of a feature vector. That means there are certain tokens in the
corpus that are most likely to appear for each class. What are they?

First, extract the feature names, the class labels, and the log probabilities
for each feature.

```{code-cell}
feature_names = tfidf_vectorizer.get_feature_names_out()
class_labels = classifier.classes_
log_probs = classifier.feature_log_prob_
```

Now iterate through every class and extract the `top_n` tokens (most probable
tokens).

```{code-cell}
:tags: [output_scroll]
top_n = 25
for idx, label in enumerate(class_labels):
    # Get the probabilities and sort them. Sorting is in ascending order, so
    # flip the array
    probs = log_probs[idx]
    sorted_probs = np.argsort(probs)[::-1]

    # The above array contains the indexes that would sort the probabilities.
    # Take the `top_n` indices, then get the corresponding tokens by indexing
    # `feature_names`
    top_probs = sorted_probs[:top_n]
    top_tokens = feature_names[top_probs]

    # Print the results
    print(f"Top tokens for {label}:")
    print("\n".join(top_tokens), end = "\n\n")
```

These are pretty general. Even with tf-idf, common tokens in fiction persist.
But we can compute the difference between log probabilities for one class and
the mean log probabilities of all other classes. That will give us more
distinct tokens for each class.

```{code-cell}
:tags: [output_scroll]
for idx, label in enumerate(class_labels):
    # Remove the current classes's log probabilities, then calculate their mean
    other_classes = np.delete(log_probs, idx, axis = 0)
    mean_log_probs = np.mean(other_classes, axis = 0)

    # Find the difference between this mean and the current class's log
    # probabilities
    difference = log_probs[idx] - mean_log_probs

    # Sort as before
    sorted_probs = np.argsort(difference)[::-1]
    top_probs = sorted_probs[:top_n]
    top_tokens = feature_names[top_probs]

    # And print
    print(f"Distinctive tokens for {label}:")
    print("\n".join(top_tokens), end = "\n\n")
```


## Visualization

Let's visualize our documents in a scatterplot so we can inspect the corpus at
scale.

### Dimensionality reduction

To do this, we'll need to transform our TF--IDF vectors into simplified
representations. Right now, these vectors are extremely **high dimensional**:

```{code-cell}
_, num_dimensions = tfidf.shape
print(f"Number of dimensions: {num_dimensions:,}")
```

This number far exceeds the two or three dimensions of plots.

We use **principal component analysis**, or PCA, to reduce the dimensionality
of our vectors so we can plot them. PCA identifies axes (principal components)
that maximize variance in data and then projects that data onto the components.
This reduces the number of features in the data but retains important
information about each vector. Take a look at Margaret Fleck's [lecture
notes][notes] if you'd like to see how this process works in detail.

[notes]: https://courses.grainger.illinois.edu/cs440/fa2019/Lectures/lect38.html

```{code-cell}
pca = PCA(0.95, random_state = 357)
pca.fit(tfidf)
```

The PCA reducer's `.explained_variance_ratio_` attribute contains the
proportion of the total variance captured by each principal component. Their
sum should equal the number we set above.

```{code-cell}
exp_variance = np.sum(pca.explained_variance_ratio_)
print(f"Explained variance: {exp_variance:.2f}%")
```

Slice out segments of these components to identify how much of the variance is
explained by the $k$-th component. `scikit-learn` sorts components
automatically, so the first ones always contain the most variance..

```{code-cell}
k = 25
exp_variance = np.sum(pca.explained_variance_ratio_[:k])
print(f"Explained variance of {k} components: {exp_variance:.2f}%")
```

The first two components do not explain very much variance, but they will be
enough for visualization.

```{code-cell}
k = 2
exp_variance = np.sum(pca.explained_variance_ratio_[:k])
print(f"Explained variance of {k} components: {exp_variance:.2f}%")
```


### Plotting documents

To plot, transform the TF--IDF scores and format the reduced data as a
DataFrame.

```{code-cell}
reduced = pca.transform(tfidf)
vis_data = pd.DataFrame(reduced[:, 0:2], columns = ["x", "y"])
vis_data["label_idx"] = corpus["hoover"].copy()
vis_data["label"] = vis_data["label_idx"].map(lambda x: periods[x])
```

Create a plot.

```{code-cell}
fig, ax = plt.subplots(figsize = (10, 10))
g = sns.scatterplot(
    x = "x",
    y = "y",
    hue = "label",
    palette = "tab10",
    alpha = 0.8,
    data = vis_data,
    ax = ax
)
g.set(title = "James chapters", xlabel = "Dim. 1", ylabel = "Dim. 2")
plt.show()
```

There isn't perfect separation here. Might some of the overlapping points be
mis-classified documents? We run predictions across all documents and re-plot
with those.

```{code-cell}
all_preds = classifier.predict(tfidf)
```

Where are labels incorrect?

```{code-cell}
vis_data["incorrect"] = np.where(
    all_preds == vis_data["label_idx"], False, True
)
```

Re-plot.

```{code-cell}
fig, ax = plt.subplots(figsize = (10, 10))
g = sns.scatterplot(
    x = "x",
    y = "y",
    hue = "label",
    style = "incorrect",
    size = "incorrect",
    sizes = (300, 35),
    palette = "tab10",
    data = vis_data,
    ax = ax,
    legend = "full"
)
g.set(title = "James chapters", xlabel = "Dim. 1", ylabel = "Dim. 2")
plt.show()
```

It does indeed seem to be the case that mis-classified documents sit right
along the border of two classes. Though keep in mind that dimensionality
reduction often results in visual distortions, so looking at data might
sometimes be misleading.

Finally, which documents are these?

```{code-cell}
idx = vis_data[vis_data["incorrect"] == True].index
model_pred = all_preds[idx]
corpus.loc[idx, ["novel", "chapter", "hoover"]].assign(model_pred = model_pred)
```
