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

Bidirectional Encoder Representations from Transformers (BERT)
==============================================================

## Preliminaries

```{code-cell}
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from transformers import pipeline
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import nltk
from nltk.corpus import stopwords
```

Load data

```{code-cell}
blurbs = pd.read_parquet("data/datasets/ltg_book_blurbs.parquet")
```

Currently the labels for this data are string representations of genres.

```{code-cell}
blurbs["d1"].sample(5).tolist()
```

We need to convert those strings into unique identifiers. In most cases, the
unique identifier is just an arbitrary number; we create them below by taking
the index position of a label in the `.unique()` output. Under the hood, the
model will use those numbers, but if we associate them in a dictionary with the
original strings, we can also have it display the original strings.

```{code-cell}
enumerated = list(enumerate(blurbs["d1"].unique()))
id2label = {idx: genre for idx, genre in enumerated}
label2id = {genre: idx for idx, genre in enumerated}
```

Use `.replace()` to remap the labels in the data.

```{code-cell}
blurbs["label"] = blurbs["d1"].replace(label2id)
```

How many unique labels are there?

```{code-cell}
num_labels = blurbs["label"].nunique()
print(num_labels, "unique labels")
```

What is the distribution of labels like?

```{code-cell}
blurbs.value_counts("label")
```

With model-ready labels made, we create a `Dataset`. These objects work
directly with the Hugging Face training pipeline to handle batch processing and
other such optimizations in an automatic fashion. They also allow you to
interface directly with Hugging Face's cloud-hosted data, though we will only
use local data for this fine-tuning run.

We only need two columns from our original DataFrame: the text and its label.

```{code-cell}
dataset = Dataset.from_pandas(blurbs[["text", "label"]])
dataset
```

Finally, we load a model to fine-tune. This works just like we did earlier,
though the `AutoModelForSequenceClassification` object expects to have an
argument that specifies how many labels you want to train your model to
recognize.

```{code-cell}
:tags: [remove-stderr]
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels = num_labels
)
```

Don't forget to associate the label mappings!

```{code-cell}
model.config.id2label = id2label
model.config.label2id = label2id
```


## Data Preparation

Data preparation will also work very much like past modeling tasks. Below, we
define a simple tokenization function. This just wraps the usual functionality
that a tokenizer would do, but keeping that functionality stored in a custom
wrapper like this allows us to cast, or **map**, that function across the
entire Dataset all at once.

```{code-cell}
def tokenize(examples):
    """Tokenize strings.

    Parameters
    ----------
    examples : dict
        Batch of texts

    Returns
    -------
    tokenized : dict
        Tokenized texts
    """
    tokenized = tokenizer(examples["text"], truncation = True)

    return tokenized
```

Now we split the data into separate train/test datasets...

```{code-cell}
split = dataset.train_test_split()
split
```

...and tokenize both with the function we've written. Note the `batched`
argument. It tells the Dataset to send batches of texts to the tokenizer at
once. That will greatly speed up the tokenization process.

```{code-cell}
:tags: [remove-stderr]
trainset = split["train"].map(tokenize, batched = True)
testset = split["test"].map(tokenize, batched = True)
```

Tokenizing texts like this creates the usual output of token ids, attention
masks, and so on:

```{code-cell}
trainset
```

Recall from the last chapter that models require batches to have the same
number of input features. If texts are shorter than the total feature size, we
pad them and then tell the model to ignore that padding during processing. But
there may be cases where an entire batch of texts is substantially padded
because all those texts are short. It would be a waste of time and computing
resources to process them with all that padding.

This is where the `DataCollatorWithPadding` comes in. During training it will
dynamically pad batches to the maximum feature size for a given batch. This
improves the efficiency of the training process.

```{code-cell}
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
```


## Model Training

With our data prepared, we move on to setting up the training process.


### Logging metrics

It's helpful to monitor how a model is doing while it trains. The function
below computes metrics when the model pauses to perform an evaluation. During
evaluation, the model trainer will call this function, calculate the scores,
and display the results.

The scores are simple ones: accuracy and F1. To calculate them, we use the
`evaluate` package, which is part of the Hugging Face ecosystem. 

```{code-cell}
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(evaluations):
    """Compute metrics for a set of predictions.

    Parameters
    ----------
    evaluations : tuple
        Model logits/label for each text and texts' true labels

    Returns
    -------
    scores : dict
        The metric scores
    """
    # Split the model logits from the true labels
    logits, references = evaluations

    # Find the model prediction with the maximum value
    predictions = np.argmax(logits, axis = 1)

    # Calculate the scores
    accuracy = accuracy_metric.compute(
        predictions = predictions, references = references
    )
    f1 = f1_metric.compute(
        predictions = predictions,
        references = references,
        average = "weighted"
    )

    # Wrap up the scores and return them for display during logging
    scores = {"accuracy": accuracy["accuracy"], "f1": f1["f1"]}

    return scores
```


### Training hyperparameters

There are a large number of hyperparameters to set when training a model. Some
of them are very general, some extremely granular. This section walks through
some of the most common ones you will find yourself adjusting.

First: epochs. The number of **epochs** refers to the number of times a model
passes over the entire dataset. Big models train for dozens, even hundreds of
epochs, but ours is small enough that we only need a few

```{code-cell}
num_train_epochs = 15
```

Training is broken up into individual **steps**. A step refers to a single
update of the model's parameters, and each step processes one batch of data.
**Batch size** determines how many samples a model processes in each step.

Batch size can greatly influence training performance. Larger batch sizes tend
to produce models that struggle to generalize (see [here][stack] for a
discussion of why). You would think, then, that you would want to have very
small batches. But that would be an enormous trade-off in resources, because
small batches take longer to train. So, setting the batch size ends up being a
matter of balancing these two needs.

[stack]: https://stats.stackexchange.com/questions/164876/what-is-the-trade-off-between-batch-size-and-number-of-iterations-to-train-a-neu

A good starting point for batch sizes is 32-64. Note that models have separate
size specifications for the training batches and the evaluation batches. It's a
good idea to keep the latter set to a smaller size, for the very reason about
measuring model generalization above.

```{code-cell}
per_device_train_batch_size = 32
per_device_eval_batch_size = 8
```

**Learning rate** controls how quickly your model fits to the data. One of the
most important hyperparameters, it is the amount by which the model updates its
weights at each step. Learning rates are often values between 0.0 and 1.0.
Large learning rates will speed up training but lead to sub-optimally fitted
models; smaller ones require more steps to fit the model but tend to produce a
better fit (though there are cases where they can force models to become stuck
in local minima).

Hugging Face's trainer defaults to `5e-5` (or 0.00005). That's a good starting
point. A good lower bound is `2e-5`; we will use `3e-5`.

```{code-cell}
learning_rate = 3e-5
```

Early in training, models can make fairly substantial errors. Adjusting for
those errors by updating parameters is the whole point of training, but making
adjustments too quickly could lead to a sub-optimally fitted model. **Warm up
steps** help stabilize a model's final parameters by gradually increasing the
learning rate over a set number of steps.

It's typically a good idea to use 10% of your total training steps as the step
size for warm up.

```{code-cell}
warmup_steps = (len(trainset) / per_device_train_batch_size) * num_train_epochs
warmup_steps = round(warmup_steps * 0.1)
print("Number of warm up steps:", warmup_steps)
```

**Weight decay** helps prevent overfitted models by keeping model weights from
growing too large. It's a penalty value added to the loss function. A good
range for this value is `1e-5` - `1e-2`; use a higher value for smaller
datasets and vice versa.

```{code-cell}
weight_decay = 1e-2
```

With our primary hyperparameters set, we specify them using a
`TrainingArguments` object. There are only a few other things to note about
initializing our `TrainingArgumnts`. Besides specifying an output directory and
logging steps, we specify when the model should evaluate itself (after every
epoch) and provide a criterion (loss) for selecting the best performing model
at the end of training. 

```{code-cell}
training_args = TrainingArguments(
    output_dir = "data/bert_blurb_classifier",
    num_train_epochs = num_train_epochs,
    per_device_train_batch_size = per_device_train_batch_size,
    per_device_eval_batch_size = per_device_eval_batch_size,
    learning_rate = learning_rate,
    warmup_steps = warmup_steps,
    weight_decay = weight_decay,
    logging_steps = 100,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "loss",
    save_total_limit = 3,
    push_to_hub = False
)
```


### Model training

Once all the above details are set, we initialize a `Trainer` and supply it
with everything we've created: the model and its tokenizer, the data collator,
training arguments, training and testing data, and the function for computing
metrics. The only thing we haven't seen below is the `EarlyStoppingCallback`.
This combats overfitting. When the model doesn't improve after some number of
epochs, we stop training.

```{code-cell}
trainer = Trainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = data_collator,
    args = training_args,
    train_dataset = trainset,
    eval_dataset = testset,
    compute_metrics = compute_metrics,
    callbacks = [EarlyStoppingCallback(early_stopping_patience = 3)]
)
```

Time to train!

```py
trainer.train()
```

Calling this method would quick off the training process, and you would see
logging information as it runs. But for reasons of time and computing
resources, the underlying code of this chapter won't run a fully training loop.
Instead, it will load a separately trained model for evaluation.

But before that, we show how to save the final model:

```py
trainer.save_model("data/models/bert_blurb_classifier/final")
```

Saving the model will save all the pieces you need when using it later.


## Model Evaluation

We will evaluate the model in two ways, first by looking at classification
accuracy, then token influence. To do this, let's re-load our model and
tokenizer. This time we specify the path to our local model.

```{code-cell}
fine_tuned = "data/models/bert_blurb_classifier/final"
tokenizer = AutoTokenizer.from_pretrained(fine_tuned)
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned)
```


### Using a pipeline

While we could separately tokenize texts and feed them through the model, a
`pipeline` will take care of all this. All we need to do is specify what kind
of task our model has been trained to do.

```{code-cell}
classifier = pipeline(
    "text-classification", model = model, tokenizer = tokenizer
)
```

Below, we put a single text through the pipeline. It will return the model's
prediction and a confidence score.

```{code-cell}
sample = blurbs.sample(1)
result ,= classifier(sample["text"].item())
```

What does the model think this text is?

```{code-cell}
print(f"Model label: {result["label"]} ({result["score"]:.2f}% conf.)")
```

What is the actual label?

```{code-cell}
print("Actual label:", sample["d1"].item())
```

Here are the top three labels for this text:

```{code-cell}
classifier(sample["text"].item(), top_k = 3)
```

Set `top_k` to `None` to return all scores.

```{code-cell}
classifier(sample["text"].item(), top_k = None)
```


### Classification accuracy

Let's look at a broader sample of texts and appraise the model's performance.
Below, we take 250 blurbs and send them through the pipeline. A more fastidious
version of this entire training setup would have spliced off this set of blurbs
before doing the train/test split. That would have kept the model from ever
seeing them until the moment we appraise performance to render a completely
unbiased view of model performance. But for the purposes of demonstration, it's
okay to sample from our data generally.

```{code-cell}
sample = blurbs.sample(250)
predicted = classifier(sample["text"].tolist(), truncation = True)
```

Now, we access the predicted labels and compare them against the true labels
with `classification_report()`.

```{code-cell}
y_true = sample["d1"].tolist()
y_pred = [prediction["label"] for prediction in predicted]
report = classification_report(y_true, y_pred, zero_division = 0.0)
print(report)
```

Overall, these are pretty nice results. The F1 scores are fairly well balanced.
Though it looks like the model struggles with classifying Biography \& Memoir
and Literary Fiction. But other genres, like Cooking and Romance, are just
fine. We can use a **confusion matrix** to see which of these genres the model
confuses with others.

```{code-cell}
confusion = confusion_matrix(y_true, y_pred)
confusion = pd.DataFrame(
    confusion, columns = label2id.keys(), index = label2id.keys()
)
```

Plot the matrix as a heatmap:

```{code-cell}
fig, ax = plt.subplots(figsize = (5, 5))
g = sns.heatmap(confusion, annot = True, cmap = "Blues", ax = ax);
ax.set(ylabel = "True label", xlabel = "Predicted label")
plt.show()
```

For this testing set, it looks like the model sometimes mis-classifies
Biography \& Memoir as Religion \& Philosophy. Likewise, it sometimes assigns
Politics to Biography \& Memoir. Finally, there appears to be a little
confusion between Literary Fiction and Romance.


### Feature analysis: SHAP values

```py
pipe = pipeline(
    "text-classification",
    model = model,
    tokenizer = tokenizer,
    top_k = None,
    truncation = True
)
# RED: positive impact, BLUE: negative impact
explainer = shap.Explainer(pipe)
```

```{code-cell}
shap_values = pd.read_parquet("data/datasets/ltg_book_blurbs_1k-shap.parquet")
shap_values.head()
```

```{code-cell}
drop_tokens = stopwords.words("english")
drop_tokens += '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“”’—–…'.split()

mask = shap_values.index.get_level_values(2).str.lower().isin(drop_tokens)
shap_values = shap_values[~mask]
```

```{code-cell}
# Initialize a DataFrame to store token-genre counts
counts = pd.DataFrame(
    0,
    index = shap_values.index.get_level_values(2).unique(),
    columns = shap_values.columns
)

# For every genre label...
for col in shap_values.columns:
    # Get the token with the highest SHAP value in each document
    max_tokens = shap_values[col].groupby(level = ["document_id"]).idxmax()

    # Format to a Series and count the number of times each token appears
    max_tokens = max_tokens.apply(pd.Series)
    max_tokens.columns = ["document_id", "token_id", "text"]
    token_counts = max_tokens.value_counts("text")

    # Index our counts DataFrame and assign the token counts
    counts.loc[token_counts.index, col] = token_counts.values
```

```{code-cell}
k = 25
topk = pd.DataFrame("", index = range(k), columns = shap_values.columns)
for col in shap_values.columns:
    tokens = counts[col].nlargest(k).index.tolist()
    topk.loc[:, col] = tokens
```

```{code-cell}
topk
```
