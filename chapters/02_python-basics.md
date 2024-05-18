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

os.chdir("..")
```


Python Basics
=============

## Preliminaries

```{code-cell}
import re
from collections import Counter
```


## Loading Data

Loading a file into your Python environment involves writing a path to the
file's location. Below, we assign a path to Mary Shelley's _Frankenstein_,
which currently sits in `data/`, a subdirectory of our current working
directory.

```{code-cell}
path = "data/shelley_frankenstein.txt"
```

Use `open` to open a connection to the file. This function requires you to
specify a value to the `mode` argument. We use `r` because are working with
**plain text** data; `rb` is for **binary** data.

```{code-cell}
fin = open(path, mode = "r") 
```

With the connection established, read the data and assign it to a new variable:

```{code-cell}
frankenstein = fin.read()
```

Finally, close the file connection:

```{code-cell}
fin.close()
```

You can accomplish these operations with less lines of code using the `with
open` pattern. There's no need to close the file connection once you've read
the data. Using this pattern, Python will do it for you.

```{code-cell}
with open(path, mode = "r") as fin:
    frankenstein = fin.read()
```

Python represents plain text files as streams of characters. That is, every
keystroke you would use to type out a text corresponds to a character. Calling
`len`, or length, on our data makes this apparent:

```{code-cell}
len(frankenstein)
```

The Penguin edition of _Frankenstein_ is about 220 pages. Assuming each page
has 350 words, that would put the book's word count in the neighborhood of
77,000 words---far less than the number above. But we see this number because
Python is counting characters, not words.


## Data Structures

Usually, however, we'll want to work with words. This requires us to change how
Python represents our text data. And, because there is no inherent concept of
what a word is in the language, it falls on us to define how to make words out
of characters. This process is called **tokenization**. Tokenizing a text means
breaking its continuous sequence of characters into separate substrings, or
tokens. There are many different ways to do this, but for now we start with a
very simple approach: we break the character sequence along whitespace
characters, characters like `\s`, for space, but also `\n` (for new lines),
`\t` (for tab), and so on.

Use the `.split()` method to break `frankenstein` along whitespace characters:

```{code-cell}
tokens = frankenstein.split()
```


### Lists

The result is a **list**, a general-purpose container for storing data. Lists
are probably the most common data structure in Python. They make very little
assumptions about the kind of data they store, and they store this data in an
ordered manner. That is, lists have a first element, a second element, and so
on up until the full length of the list.

```{code-cell}
len(tokens)
```

To select an element, or a group of elements, from a list, you **index** the
list. The square brackets `[]` are Python's index operator. Use them in
conjunction with the **index position** of the element(s) you want to select.
The index position is simply a number that corresponds to where in the list an
element is located.

```{code-cell}
tokens[42]
```

Python uses **zero-based indexing**. That means the positions of elements are
counted from 0, not 1.

```{code-cell}
tokens[0]
```

Use the colon `:` to select multiple elements.

```{code-cell}
tokens[10:20]
```

Setting no starting position takes all elements in the list up to your index
position:

```{code-cell}
tokens[:10]
```

While leaving off the ending position takes all elements from an index position
to the end of the list:

```{code-cell}
tokens[74970:]
```

Alternatively, count backwards from the end of a list with a negative number:

```{code-cell}
tokens[-5:]
```

Add another colon to take every n-th element in your selection. Below, we take
every second element from index positions 100-200.

```{code-cell}
:tags: [output_scroll]
tokens[100:200:2]
```

Leave the first selection unspecified to take every n-th element across the
whole list:

```{code-cell}
:tags: [output_scroll]
tokens[::1000]
```

You can also use `[]` to create a list manually. Here's an empty list:

```{code-cell}
[]
```

And here is one with inhomogeneous data. That is, it contains all kinds of data
types---and another lists! Lists can contain lists.

```{code-cell}
li = [8, "x", False, ["a", "b", "c"]]
```

To index an element in this sublist, you'll need to select the index position
of the sublist, then select the one for the element you want.

```{code-cell}
li[3][1]
```

You can set the element of a list by assigning a value at that index:

```{code-cell}
li[2] = True
li
```

Assigning elements of a container is not without complication. Below, we use
the `list` keyword---another method of creating a list---to break a character
string into individual pieces. We assign the output of this to `x`. Then, we
create a new variable, `y`, from `x`.

```{code-cell}
x = list("abc")
y = x
```

Assigning a new value to an index position in `x` will propagate the change to
`y`.

```{code-cell}
x[2] = "d"
print(x, y)
```

Why did this happen? When you create a list and assign it to a variable, the
variable points, or **refers**, to the location of the list in your computer's
memory. If you create a second variable from the first, it will refer to the
first variable, which in turn refers to the data. As a result, operations
called on one variable will affect the other, and vice versa.

When in doubt, use `.copy()` to prevent this.

```{code-cell}
x = list("abc")
y = x.copy()
x[2] = "d"
print(x, y)
```

### Tuples

References can be confusing. If you know that the elements of a container
shouldn't change, you can also avoid the problem above by creating a **tuple**.
Like a list, a tuple is a one-dimensional container. The key difference is that
tuples are **immutable**: once you create a tuple, you are neither able to
alter it or its elements.

Make a tuple by enclosing comma-separated values in parentheses `()`.

```{code-cell}
tup = (1, 2, 3)
tup
```

Alternatively, convert a list to a tuple using `tuple`.

```{code-cell}
x = list("abc")
x = tuple(x)
x
```

You will get an error if you attempt to change this tuple.

```{code-cell}
:tags: [raises-exception]
x[2] = "d"
```


### Sets

Unlike lists and tuples, **sets** cannot contain multiple instances of the same
element. Create them using curly brackets `{}` or the `set` keyword.

```{code-cell}
set_a = {"a", "b", "c"}
set_b = set("aabc")
set_a == set_b
```

Sets are useful containers for keeping track of features in your data. For
example, converting our list of tokens to a set will automatically prune out
all repeated tokens. The result will be a set of in NLP are called **types**:
the unique elements in a document. In effect, it is the vocabulary of the
document.

```{code-cell}
types = set(tokens)
print("Number of types:", len(types))
```

Sets also offer additional functionality for performing comparisons. We won't
touch on this too much in the following chapters, but it's useful to know
about. Given the following two sentences from _Frankenstein_, for example:

```{code-cell}
a = "I am surrounded by mountains of ice which admit of no escape and threaten every moment to crush my vessel."
b = "This ice is not made of such stuff as your hearts may be; it is mutable and cannot withstand you if you say that it shall not." 
```

We convert both to sets:

```{code-cell}
a = set(a.split())
b = set(b.split())
```

We can find their **intersection**. This is where the two sentences'
vocabularies overlap:

```{code-cell}
a.intersection(b)
```

We can also find their **difference**:

```{code-cell}
a.difference(b)
```

Finally, we can build a new set that combines the two:

```{code-cell}
:tags: [output_scroll]
c = a.union(b)
c
```

The downside of sets, however, is that they are unordered. This means they
cannot be indexed.

```{code-cell}
:tags: [raises-exception]
c[5]
```


### Dictionaries

Finally, there are **dictionaries**. Like sets, dictionaries store unique
elements, but they associate those elements with a particular value (these can
be individual values, like numbers, or containers). Every element in a
dictionary is therefore a **key--value pair**. This makes them powerful data
structures for associating values in your data with metadata of one kind or
another.

Create a dictionary with curly brackets `{}` and colons `:` that separate the
key--value pairs.

```{code-cell}
counts = {"x": 4, "y": 1, "z": 6}
counts
```

Unlike sets, dictionaries can be indexed by their keys. This returns the value
stored at a particular key.

```{code-cell}
counts["y"]
```

Assign a new value to a key to update it.

```{code-cell}
counts["z"] = counts["z"] - 1
counts
```

Or use the `.update()` method in conjunction with the curly bracket and colon
syntax. Note that this is an **in place** operation. You do not need to
reassign the result to a variable.

```{code-cell}
counts.update({"x": 10})
counts
```

Either method also enables you to add new keys to a dictionary.

```{code-cell}
counts["w"] = 7
counts
```

Use the `.keys()` method to get all keys in a dictionary (this is functionally
a set):

```{code-cell}
counts.keys()
```

And use the `.values()` method to get values:

```{code-cell}
counts.values()
```

At the beginning of the chapter we imported a `Counter` object. This is a
special kind of dictionary. It counts its input and stores the results as
key--value pairs.

A `Counter` works on characters:

```{code-cell}
:tags: [output_scroll]
Counter(frankenstein)
```

But it also works on containers like lists. That makes them highly useful for
our purposes. Below we calculate the token counts in _Frankenstein_...

```{code-cell}
token_freq = Counter(tokens)
```

...and use the `.most_common()` method to get the top-10 most frequent tokens
in the novel:

```{code-cell}
top_ten = token_freq.most_common(10)
top_ten
```


## Iteration

The output of `.most_common()` is a list of tuples. You'll see patterns like
this frequently: data structures wrapping other data structures. But while we
could work with this list as we could any list, indexing it to retrieve tuples,
which we could then index again, that would be inefficient for many operations.
More, it might require us to know in advance which elements are at what index
positions.

It would be better to work with our data in a more programmatic fashion. We can
do this with the above containers because they are all **iterables**: that is,
they enable us to step through each of their elements and do things like
perform checks, run calculations, or even move elements to other parts of our
code. This is called **iterating** through our data; each step is one
**iteration**.


### `for` loops

The standard method for advancing through an iterable is a `for` loop. Even if
you've never written a line of code before, you've probably heard of them. A
`for` loop begins with the `for` keyword, followed by:

+ A placeholder variable, which will be automatically assigned to an element at
  the beginning of each iteration
+ The `in` keyword
+ An object with elements
+ A colon `:`

Code in the body of the loop must be indented by 4 spaces.

Below, we iterate through each tuple in `top_ten`:

```{code-cell}
:tags: [output_scroll]
for tup in top_ten:
    print(tup)
```

Within the indented portion of the `for` loop, you can perform checks and
computations. In every iteration below, we assign the token in the tuple to a
variable `tok` and its value to `val`. Then, we check whether `val` is even. If
it is, we print `tok` and `val`.

```{code-cell}
:tags: [output_scroll]
for tup in top_ten:
    tok = tup[0]
    val = tup[1]
    if val % 2 == 0:
        print(tok, val)
```

Oftentimes you want to save the result of a check. The easiest way to do this
is by creating a new, empty list and using `.append()` to add elements to it.

```{code-cell}
is_even = []
for tup in top_ten:
    val = tup[1]
    if val % 2 == 0:
        is_even.append(tup)

print(is_even)
```


### `while` loops


### Comprehensions
