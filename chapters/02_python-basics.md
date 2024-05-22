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

This chapter overviews key functionality and concepts in Python. We will learn
how to load files into Python, store data in various data structures, and
iterate through that data. Additionally, we will discuss regular expressions
for string matching and write our own functions.

+ **Data**: a plain text version of Mary Shelley's _Frankenstein_


## Preliminaries

To do our work, we will import an entire module and a single object from
another module. 

```{code-cell}
import re
from collections import Counter
```

Recall from the last chapter that importing an entire module gives us access to
all of its functionality. We specify which part of that module we want to use
with dot `.` notation.

```{code-cell}
re.findall
```

To use the object we imported, initialize it by assigning it to a variable:

```{code-cell}
c = Counter()
c
```


## Loading Data

Loading a file into your Python environment requires writing a path to the
file's location. Below, we assign a path to Mary Shelley's _Frankenstein_,
which currently sits in `data/`, a subdirectory of our current working
directory.

```{code-cell}
path = "data/shelley_frankenstein.txt"
```

Use `open` to open a connection to the file. This function requires you to
specify a value to the `mode` argument. We use `r` because are working with
**plain text** data; `rb` would for **binary** data.

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

Usually, however, we want to work with words. This requires us to change how
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

The result of `.split()` is a **list**, a general-purpose, one-dimensional
container for storing data. Lists are probably the most common data structure
in Python. They make very little assumptions about the kind of data they store,
and they store this data in an ordered manner. That is, lists have a first
element, a second element, and so on up until the full length of the list.

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

And here is one with all kinds of data types---and another list! Lists can
contain lists.

```{code-cell}
l = [8, "x", False, ["a", "b", "c"]]
```

To index an element in this sublist, you'll need to select the index position
of the sublist, then select the one for the element you want.

```{code-cell}
l[3][1]
```

You can set the element of a list by assigning a value at that index:

```{code-cell}
l[2] = True
l
```

Assigning elements of a container is not without complication. Below, we use
the `list` keyword---another method of creating a list---to break a character
string into individual pieces. We assign the output of this to `x`. Then, we
create a new variable, `y`, from `x`.

```{code-cell}
x = list("abc")
y = x
y
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
Like a list, a tuple is a one-dimensional container for general data storage.
The key difference is that tuples are **immutable**: once you create a tuple,
you are neither able to alter it nor its elements.

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
element. They only have unique elements. Create them using curly brackets `{}`
or the `set` keyword.

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

We split them into tokens and convert both to sets:

```{code-cell}
a = set(a.split())
b = set(b.split())
```

Now, we find their **intersection**. This is where the two sentences'
vocabularies overlap:

```{code-cell}
a.intersection(b)
```

We can also find their **difference**, or the set of tokens that do not
overlap:

```{code-cell}
a.difference(b)
```

Finally, we can build a new set that combines our two sets:

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
elements, but they associate those elements with a particular value. These can
be individual values, like numbers, or containers, like lists, tuples, and so
on. Every element in a dictionary is therefore a **key--value pair**. This
makes dictionaries powerful data structures for associating values in your data
with metadata of one kind or another.

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

The `.get()` method also works.

```{code-cell}
counts.get("y")
```

You can set a default value to control for cases when the dictionary doesn't
have a requested key.

```{code-cell}
counts.get("a", None)
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

Using `.pop()` removes a key--value pair.

```{code-cell}
counts.pop("w")
counts
```

The `.keys()` method returns all keys in a dictionary. Functionally, this is a
set.

```{code-cell}
counts.keys()
```

Alternatively the `.values()` method returns a dictionary's values.

```{code-cell}
counts.values()
```

At the beginning of the chapter we imported a `Counter` object. This is a
special kind of dictionary. It counts its input and stores the results as
key--value pairs.

A `Counter` can work on characters:

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
positions. This information is not always easily available, especially when
writing general-purpose code.

It would be better to work with our data in a more programmatic fashion. We can
do this with the above containers because they are all **iterables**: that is,
they enable us to step through each of their elements and do things like
perform checks, run calculations, or even move elements to other parts of our
code. This is called **iterating** through our data; each step is one
**iteration**.


### For-loops

The standard method for advancing through an iterable is a for-loop. Even if
you've never written a line of code before, you've probably heard of them. A
for-loop begins with the `for` keyword, followed by:

+ A placeholder variable, which will be automatically assigned to an element at
  the beginning of each iteration
+ The `in` keyword
+ An object with elements
+ A colon `:`

Code in the body of the loop must be indented. An equivalent of four spaces for
indentation is standard.

Below, we iterate through each tuple in `top_ten`. At the start of the
iteration, a tuple is assigned to `tup`; we then print this tuple.

```{code-cell}
:tags: [output_scroll]
for tup in top_ten:
    print(tup)
```

For-loops can be nested inside of for-loops. Let's re-implement the above with
two for-loops.

```{code-cell}
:tags: [output_scroll]
for tup in top_ten:
    for part in tup:
        print(part)
    print("\n")
```

See how the outer `print` statement only triggers once the inner for-loop has
finished? Every iteration of the first for-loop kicks off the second for-loop
anew.

Within the indented portion of a for-loop you can perform checks and
computations. In every iteration below, we assign the token in the tuple to a
variable `tok` and its value to `val`. Then, we check whether `val` is even. If
it is, we print `tok` and `val`.

```{code-cell}
:tags: [output_scroll]
for tup in top_ten:
    tok, val = tup[0], tup[1]
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

Other data structures are iterable in Python. In addition to lists, you'll find
yourself iterating through dictionaries with some frequency. Use `.keys()` or
`.values()` to iterate, respectively, through the keys and values of a
dictionary. Or, use `.items()` to iterate through both at the same time. Note
that `.items()` requires using two placeholder variables separated by a comma
`,`.

```{code-cell}
for key, value in counts.items():
    print(key, "->", value)
```

Below, we divide every count in `token_freq` by the total number of tokens in
_Frankenstein_ to express counts as percentages, using a new `Counter` to store
our results.

```{code-cell}
num_tokens = token_freq.total()
percentages = Counter()

for token, count in token_freq.items():
    percent = count / num_tokens
    percentages[token] = percent
```

Here is the equivalent of `top_ten`, but with percentages:

```{code-cell}
:tags: [output_scroll]
percentages.most_common(10)
```


### Comprehensions

**Comprehensions** are idiomatic to Python. They allow you to perform
operations across an iterable without needing to pre-allocate an empty copy to
store the results. This makes them both concise and efficient. You will most
frequently see comprehensions used in the context of lists (i.e. "list
comprehensions"), but you can also use them for dictionaries and sets. 

The syntax for comprehension includes the keywords `for` and `in`, just like a
for-loop. The difference is that in the list comprehension, the repeated code
comes _before_ the `for` keyword rather than after it, and the entire
expression is enclosed in square brackets `[ ]`.

Below, we use the `.istitle()` method to find capitalized tokens in
_Frankenstein_. This method returns a Boolean value, so the resultant list will
contain `True` and `False` values that specify capitalization at a certain
index.

```{code-cell}
is_title = [token.istitle() for token in tokens]
is_title[:10]
```

That should be straightforward enough, but we don't know which tokens these
values reference. With comprehensions, an easy way around this is to use an
`if` statement embedded in the comprehension. Put that statement and a
conditional check at the end of the comprehension to filter a list.

```{code-cell}
is_title = [token for token in tokens if token.istitle()]
is_title[:10]
```

Comprehensions become particularly powerful when you use them to manipulate
each element in an iterable. Below, we change all tokens to their lowercase
variants using `.lower()`.

```{code-cell}
lowercase = [token.lower() for token in tokens]
lowercase[:10]
```


### While-loops

While-loops continue iterating until a condition is met. Whereas a for-loop
only iterates through your data once, a while-loop iterates indefinitely. That
means you need to specify an **exit condition** to break out of your
while-loop, otherwise your code will get trapped and eventually your computer
will kill the process.

The syntax for a while-loop is quite simple: start it with `while` and a
condition. Below, we increment a counter to print the first ten tokens in
_Frankenstein_.

```{code-cell}
current_index = 0
while current_index < 10:
    print(tokens[current_index])

    current_index += 1
```

Note that we must specify, and then manually increment, the counter. If we
didn't, the loop would have no reference telling it when it should break.

Here is a more open-ended loop. We set the condition to `True`, keeping the
loop running until we reach an exit condition. Then, for each iteration, we
index our list of tokens and check whether the token at that index matches the
one we're looking for. If it does, the code prints that index position and
stops the iteration with a `break` statement. If it doesn't, we increment the
counter and try again.

```{code-cell}
find_first = "Frankenstein"
current_index = 0
while True:
    token = tokens[current_index]
    if token == find_first:
        print("The first occurrence of", find_first, "is at", current_index)
        break

    current_index += 1
```


## Regular Expressions

You have likely noticed by now that our tokenization strategy has created some
strange tokens. Most notably, punctuation sticks to words because there was no
whitespace to separate them. This means that, for our `Counter`, the following
two tokens are counted separately, even though they're the same word:

```{code-cell}
variants = ["ship,", "ship."]
for tok in variants:
    print(tok, "->", token_freq[tok])
```

We can handle this in a number of ways. Many rely on writing out **regular
expressions**, or regexes. Regexes are special sequences of characters that
represent patterns for matching in text; these sequences are comprised of
regular old characters in text, or **literals**, and **metacharacters**,
special characters that stand for whole classes of literals. Regexes work as a
search mechanism, and they become highly useful in text processing for their
ability to find variants like the tokens above.


### Literals

The following regex will match on the string "ship":

```{code-cell}
ship = r"ship"
```

Note how we prepend our string with `r`. That tells Python to treat the string
as a regex sequence. Using `findall()` from the `re` module will return a list
of all matches on this regex:

```{code-cell}
:tags: [output_scroll]
re.findall(ship, frankenstein)
```

When you use literals like this, Python will match only on the exact sequence.
But that's a problem for us, because there's no way to know whether the above
output refers to "ship" and any following punctuation, or if our regex has also
matched on words that contain "ship," like "relationship" and "shipment."

The latter will most certainly be the case. We'll see this if we search with
`finditer()`. It finds all matches and also returns where they start and end in
the character sequence (the object returned is a `Match`). Below, we use those
start/end positions to glimpse the context of matches.

```{code-cell}
:tags: [output_scroll]
found = re.finditer(ship, frankenstein)
for match in found:
    # Get the match text
    span = match.group()

    # Get its start and end, then offset both
    start = match.start() - 2
    end = match.end() + 2

    # Ensure our expanded start/end locations don't overshoot the string
    if start < 0:
        start = 0
    if end > len(frankenstein):
        end = len(frankenstein)

    print(span, "->", frankenstein[start:end])
```


### Metacharacters

Controlling for cases where our regex returns more than what we want requires
metacharacters.

The `.` metacharacter stands for any character except a newline `\n`.

```{code-cell}
:tags: [output_scroll]
re.findall(r"ship.", frankenstein)
```

If you want the literal period `.`, you need to use an **escape character**
`\`.

```{code-cell}
re.findall(r"ship\.", frankenstein)
```

Note that this won't work:

```{code-cell}
:tags: [raises-exception]
re.findall(r"\", frankenstein)
```

Instead, escape the escape character:

```{code-cell}
re.findall(r"\\", frankenstein)
```

No such characters in this text, however.

Use `+` as a repetition operator to find instances where the preceding
character is repeated at least once, but with no limit up to a newline
character. If we use it with `.`, it returns strings up to the ends of lines.

```{code-cell}
:tags: [output_scroll]
re.findall(r"ship.+", frankenstein)
```

Related to `+` is `?` and `*`. The first means "match zero or one", while the
second means "match zero or more". An example of `*` is below. Note we change
our regex slightly to demonstrate the zero matching.

```{code-cell}
:tags: [output_scroll]
re.findall(r"ship*", frankenstein)
```

Use curly brackets `{ }` in conjunction with numbers to specify a limit for how
many repetitions you want. Here is "match three to five":

```{code-cell}
:tags: [output_scroll]
re.findall(r"ship.{3,5}", frankenstein)
```

Want to constrain your search to particular characters? Parentheses `( )`
specify groups of characters, including metacharacters. Use them in conjunction
with the "or" operator `|` to get two (or more) variants of a string.

```{code-cell}
:tags: [output_scroll]
re.findall(r"(ship\.|ship,)", frankenstein)
```

Or, use square brackets `[ ]` to specify literals following an "or" logic, e.g.
"character X or character Y or...".

```{code-cell}
:tags: [output_scroll]
re.findall(r"ship[.,]", frankenstein)
```

Note that literals are case-sensitive.

```{code-cell}
:tags: [output_scroll]
re.findall(r"[Ss]everal", frankenstein)
```

Including a space character is valid here:

```{code-cell}
:tags: [output_scroll]
re.findall(r"ship[., ]", frankenstein)
```

But you can also use `\s`. This specifies a **character class**: whole types of
characters (in this case, spaces).

```{code-cell}
:tags: [output_scroll]
re.findall(r"ship[.,\s]", frankenstein)
````

Below, we find all spaces (and multiple space sequences) in the novel:

```{code-cell}
:tags: [output_scroll]
re.findall(r"\s+", frankenstein)
```

There are also character classes for digits `\d` and alphanumeric characters
`\w`. Here is an example with digits, which you could use to find chapter
breaks:

```{code-cell}
:tags: [output_scroll]
re.findall(r"Chapter \d+", frankenstein)
```

Using `\w`, the pattern below specifies alphanumeric characters followed by a
newline.

```{code-cell}
:tags: [output_scroll]
re.findall(r"\w+\n", frankenstein)
```

The start-of-text anchor operator `^` is useful for filtering out characters.
It checks whether a sequence begins with the characters that follow it. Below,
we select characters that are neither alphanumeric nor spaces.

```{code-cell}
:tags: [output_scroll]
re.findall(r"[^\w\s]+", frankenstein)
```

The `sub()` function will substitute regex matches with another sequence. If we
use the same pattern above, we can remove all punctuation. Note that we also
need to tack on the extra underscore character, as it is technically counted in
the character class. 

```{code-cell}
cleaned = re.sub(r"[^\w\s]+|_", " ", frankenstein)
```

This is one way of getting around those variants from above.

```{code-cell}
token_freq = Counter(cleaned.split())

variants = ["ship,", "ship.", "ship"]
for tok in variants:
    print(tok, "->", token_freq.get(tok, None))
```

We actually scooped up even more tokens from this substitution pattern. An even
better picture of our counts would emerge if we changed our text to lowercase
so that the `Counter` can count case variants together.

```{code-cell}
cleaned = cleaned.lower()
token_freq = Counter(cleaned.split())

print("Unique tokens after substitution and case change:", len(token_freq))
```

We'll leave off on text preprocessing for now but will pick it up in the next
chapter. We've covered most of the main regexes, though there are a few more
advanced ones that you may find useful. See this [cheatsheet][cheatsheet] for
an extensive overview.

[cheatsheet]: https://www.pythoncheatsheet.org/cheatsheet/regular-expressions


## Functions

So far we have relied on external functions, but we can also write our own.
Writing functions greatly reduces redundancy in code, because you can reuse
them as many times as you want, in whatever contexts you want. More, functions
can keep your code organized. In complex processes, it often helps to break
your code up into individual steps and associate each with a function. That
also makes rewriting code much easier later on.

Before we write a function, here is a review of vocabulary associated with
them:

+ The placeholder variables for inputs are **parameters**
+ **Arguments** are the values assigned to parameters during a call
+ To **call** a function means using it to compute something
+ The **body** is the code inside a function
+ A function's **scope** is the local context in which it runs code
+ The **return value** is the output of a function

In Python, a function begins with the `def` keyword, followed by:

+ The name of the function
+ A list of parameters surrounded by parentheses
+ A colon `:`

There is no practical limit to the number of parameters. Code in the body of
the function should be indented according to the same conventions for loops and
conditionals (four spaces). To return a result from the function, use the
`return` keyword.

Here is a very simple function that returns `True`/`False` depending on whether
`text` starts with `character`.

```{code-cell}
def starts_with(text, character):
    first_char = text[0]
    return first_char == character
```

Call the function by writing out its name and supplying it with arguments.

```{code-cell}
starts_with("Book", "B")
```

Here are some more examples:

```{code-cell}
starts_with("natural language processing", "w")
```

```{code-cell}
to_test = [("token", "t"), ("Character", "c")]
for testing_pair in to_test:
    text = testing_pair[0]
    character = testing_pair[1]
    text_starts_with_char = starts_with(text, character)

    print(text, "starts with", character, "is", text_starts_with_char)
```

When we speak of a function's scope, we are talking about the local context for
that function. Typically, variables inside the function or its arguments are
not the same as the ones you set elsewhere in your code, even if the names of
those variables match. 

Here is a simple example with a negative number check:

```{code-cell}
def is_negative(x):
    return x < 0
```

Now call it:

```{code-cell}
x = -4
is_negative(5)
```

You'll often find yourself transforming code you've already written into a
function. The function below re-implements the for-loop we wrote above to print
regex matches and their context. It has three parameters:

1. The regex match is `match`
2. The text where we found the match is `string`
3. Our `offset` is the number of characters to extract on either side of the
   match

```{code-cell}
def show_match_context(match, string, offset):
    # Get the match text
    span = match.group()

    # Get its start and end, then offset both
    start = match.start() - offset
    end = match.end() + offset

    # Ensure our expanded start/end locations don't overshoot the string
    if start < 0:
        start = 0
    if end > len(string):
        end = len(string)

    # Print the results
    print(span, "->", string[start:end])
```

With our function defined, we can call it once.

```{code-cell}
match = re.search(r"lightning", frankenstein)
show_match_context(match, frankenstein, 5)
```

Or as many times as we please.

```{code-cell}
:tags: [output_scroll]
found = re.finditer(r"lightning", frankenstein)
for match in found:
    show_match_context(match, frankenstein, 5)
```

It's somewhat annoying to have to write out the offset every time we call the
function. To circumvent this, you can specify a default value for a parameter.
Your function will use that if you do not supply an argument for that
parameter.

```{code-cell}
def show_match_context(match, string, offset = 5):
    # Get the match text
    span = match.group()

    # Get its start and end, then offset both
    start = match.start() - offset
    end = match.end() + offset

    # Ensure our expanded start/end locations don't overshoot the string
    if start < 0:
        start = 0
    if end > len(string):
        end = len(string)

    # Print the results
    print(span, "->", string[start:end])
```

With no argument supplied:

```{code-cell}
match = re.search(r"lightning", frankenstein)
show_match_context(match, frankenstein)
```

Supplying an argument:

```{code-cell}
match = re.search(r"lightning", frankenstein)
show_match_context(match, frankenstein, 15)
```

Recall that the output of `finditer()` is a special `Match` object, which has
properties that extend beyond typical strings. If you forget this and try to
call your function on an object it doesn't expect, you'll run into an error:

```{code-cell}
:tags: [raises-exception]
show_match_context("lightning", frankenstein)
```

The more code you write, the harder it is to keep this sort of thing in your
mind. That's why it's helpful to document your function with a **docstring**.
Docstrings are descriptions of what your function does and what kinds of
parameters it expects. They go on the first line of your function's body and
are surrounded by triple quotes `"""`.

```{code-cell}
def starts_with(text, character):
    """Determine whether a string starts with a character."""
    first_char = text[0]
    return first_char == character
```

Once you've written a docstring, you can use `help` in a Python console or `?`
in a Jupyter Notebook to display this information.

```{code-cell}
help(starts_with)
```

There are several styles for writing docstrings, but the NumPy
[conventions][style] are good ones. They specify docstrings like so:

```py
def func(x, y):
    """A short summary description of a function ending with a period.
    
    A longer description if necessary.

    Parameters
    ----------
    x : x's datatype
        A description of what x is
    y : y's datatype
        A description of what y is

    Returns
    -------
    value : value's datatype
        A description of what value is (only supply if the function returns a
        value)
    """
```

Let's document `show_match_context` with a docstring.

```{code-cell}
def show_match_context(match, string, offset = 5):
    """Print a regex match's surrounding characters.

    Parameters
    ----------
    match : re.Match
        A regex match from re.search() or re.finditer()
    string : str
        The string in which the match was found
    offset : int
        The number of surrounding characters to the left/right of match
    """
    # Get the match text
    span = match.group()

    # Get its start and end, then offset both
    start = match.start() - offset
    end = match.end() + offset

    # Ensure our expanded start/end locations don't overshoot the string
    if start < 0:
        start = 0
    if end > len(string):
        end = len(string)

    # Print the results
    print(span, "->", string[start:end])
```

Now our function is fully documented and ready for later use.

```{code-cell}
help(show_match_context)
```
