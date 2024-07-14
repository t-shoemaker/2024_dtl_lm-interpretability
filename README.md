# Introduction to Interpretability for Language Models

_Instructor:_ Tyler Shoemaker
_Dates:_ 07/15/2014--08/02/2024
_Meeting time:_ MWF, 12:00-3:00pm EST

This is a three-week crash course on interpretability and language modeling.
The course starts with an introduction to Python, moves on to language models
in natural language processing, and ends with a week on large language models
like BERT and the GPT series.


## Syllabus

| Day | Title                   | Topics                                                          |
|-----|-------------------------|-----------------------------------------------------------------|
|  1  | Getting Started         | terminal, environment managment, Python introduction            |
|  2  | Python Basics           | control flow, data structures, functions                        |
|  3  | Data Analysis in Python | tabular data, plotting, corpus analytics                        |
|  4  | N-gram Models           | n-grams, probability models, sampling                           |
|  5  | Vectorization           | the document-term matrix, weighting, classification             |
|  6  | Vector Space Semantics  | semantic spaces, vector operations, static embeddings           |
|  7  | Introduction to LLMs    | subwords, model architectures, dynamic embeddings               |
|  8  | BERT                    | fine tuning, model evaluation, SHAP values                      |
|  9  | GPT                     | next token prediction, reverse engineering, activation patching |


## Data

As of this writing (July 2024), a zipped data directory for the course,
`dtl_2024.zip`, may be found at [tylershoemaker.info/data][data]. Download this
file, move it to the location on your computer where you'll be working from,
and unzip it.

[data]: https://tylershoemaker.info/data


## Rendering the Reader

To render this reader, follow these steps:

1. Download the data (instructions above) and place the unzipped directory in
   the root of this repository
2. Create the `conda`/`mamba` environment:
   ```sh
   <conda/mamba/micromamba> env create --file env.yml
   ```
3. Build the book:
   ```sh
   jupyter-book build .
   ```
4. Push to the `gh-pages` branch:
   ```sh
   ghp-import -n -p -f _build/html
   ```
