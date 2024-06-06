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

Getting Started
===============

This chapter introduces fundamental concepts for working with code. We begin
with the Unix command line and environment management, after which we will turn
to an overview of the Python programming language.

+ **Prerequisites:** if you are using Windows, install [Git
for Windows][git]. It comes with Git Bash, a Unix-style command-line interface.

[git]: https://git-scm.com/downloads


## Command Line Basics

Most modern coding platforms follow the conventions of mainstream computing,
using a **graphical user interface** (GUI) to display information and accept
user input. But there is still real value in knowing the basics of the
**command-line interface** (CLI), which relies solely on text input. Learning
how to use the CLI will teach you a great deal about how your computer works,
and there are still several instances where CLIs are the only point of access
to a system.

The CLI contains three main parts:

1. A **terminal**: an environment you use to send/receive information to/from
   your computer
2. A **shell**: a program that executes commands you enter into the terminal
3. The **command line**: where you enter the commands that the shell executes

As we move through this chapter, keep one key thing in mind: a CLI is just
another mode of interaction with your computer. Everything you can access
through a windowing system on a GUI is also accessible with a CLI.


### Launching a command-line interface

MacOS comes with a pre-installed CLI called Terminal. To launch it, go to:

```
Applications -> Utilities -> Terminal
```

Windows users will use Git Bash. To launch it, do the following:

```
Windows Start Menu -> search "Git Bash"
```

Or:

```
Windows Start Menu -> Programs -> Git Bash
```

You should see something like this when your CLI launches:

![A command-line interface window](../img/cli.png)

The window above shows a basic command line with a **prompt**. A prompt is an
input field where you type your commands, and it is prepended with useful
information about your computer:

+ First, `tyler@brain` tells us the current user (`tyler`) and the current 
  computer (`brain`). That may seem like redundant information, but with the
  command line it's possible to log in to other computers
+ The prompt also tells us where we are on this computer's file system. Right
  now we are at `~`. We'll discuss what this symbol means later. For now, just
  keep in mind that this is where you should look to get your bearings
+ Finally, we see `%`. It is an indicator that the CLI is is waiting for input

```{tip}
In the following sections, we will display CLI commands by prepending them with
an indicator character, `$`. You do not need to type this character yourself,
it's just to signal that a cell shows a command.
```


### Input and interactions

Time to enter some commands. This is simply a matter of typing them in after
the prompt and pressing `Return`/`Enter`.

For example, `echo` will print back to screen any text that you supply the
command:

```
$ echo "Hello world!"
Hello world!
```

You can see what files and folders are in your current location with `ls`
("list"):

```
$ ls
_build      _config.yml _toc.yml    chapters    data        img         src
```

Send `ls` the name of a subfolder to show its contents:

```
$ ls chapters
01_getting-started.md      04_ngram-models.md         07_intro-to-llms.md        index.md
02_python-basics.md        05_vectorization.md        08_bert.md
03_data-analysis-basics.md 06_vector-spaces.md        09_gpt.md
```

Want more information? Modify the base functionality of `ls`---or any
command---with **flags**. We do this by adding a dash `-` and then a letter, or
a combination of letters, directly after the dash. Below, we send `ls` flags to
have it show all contents (`a`) with long-formatted output (`l`) in a
human-readable fashion (`h`):

```
$ ls -alh
drwxr-xr-x  13 <username>  staff   416B May 25 01:46 .
drwxr-xr-x   5 <username>  staff   160B May  8 12:58 ..
drwxr-xr-x  14 <username>  staff   448B May 25 01:35 .git
-rw-r--r--   1 <username>  staff    20B Apr 22 15:44 .gitignore
drwxr-xr-x   6 <username>  staff   192B Apr 25 22:01 _build
-rw-r--r--   1 <username>  staff   501B Apr 21 23:23 _config.yml
-rw-r--r--   1 <username>  staff   583B Apr 22 10:10 _toc.yml
drwxr-xr-x  12 <username>  staff   384B May 30 13:00 chapters
drwxr-xr-x  11 <username>  staff   352B May 17 17:16 data
drwxr-xr-x   5 <username>  staff   160B May 30 12:37 img
drwxr-xr-x   5 <username>  staff   160B May 25 01:35 src
```

Long-formatted output shows you metadata about files and folders, including
information about the their owner/group (`<username>` and `staff`), their
permissions (those strings of `r`'s, `w`'s, and `x`'s), their size, and the
last date they were modified. We won't discuss permissions in depth, but know
that they dictate who can read `r`, write `w`, and execute `x` files.

Any file or folder prepended with `.` in its name will be hidden by default on
your computer. But using the all `-a` flag, as we did above, will show them:
`.`, `..`, `.git`, and `.gitignore`. The first two are special notations for
navigating your computer, which we discuss in the next section. The third,
`.git`, is a subfolder that stores version control information about this
reader, and `.gitignore` is a **dotfile**. Dotfiles often contain various
configuration settings that people use to customize their computers. In this
case, `.gitignore` controls what parts of this folder version control should
ignore.


### Command-line syntax

Depending on your computer and CLI, the above output may differ slightly, but
in general the basic presentation and functionality will be the same. Commands
use a space to delimit their different components, and flags are called with
`-` to modify those commands. When put together, we can generalize these
components to a command-line **syntax**:

```
$ <command> <optional flags> <file/data on which to run the command>
```

One caveat: since spaces are a meaningful part of the command-line syntax, file
names with spaces in them can cause problems. You'll need to **escape** these
names with `\`.

This will throw an error:

```
$ <command> file name.txt
```

But this will not:

```
$ <command> file\ name.txt
```

Be sure to read error messages when you see them. While the CLI is often quite
mute, it will do its best to show you what you did wrong when an error arises.
Below, for example, the shell explains that it cannot find a command, which
actually stems from a typing error:

```
$ lschapters
zsh: command not found: lschapters
```

Likewise, here, `ls` cannot find the requested folder because it does not
exist:

```
$ ls storage
ls: storage: No such file or directory
```

One last thing about general CLI usage. Sometimes, you need to stop a process
immediately. Use `CTRL+C` to **interrupt** it. But keep in mind that, for the
most part, it isn't possible to undo a command. Take care to know exactly what
you're running and what you're running it on, especially as you get acclimated
to the CLI.


### Getting help

There are dozens of commands available to you on your computer, and you can
install even more. If you'd like to see an overview of some of the most common
commands, take a look at Valerie Summet's [Unix cheat sheet][sheet].

[sheet]: https://geog-464.github.io/unix_cheatsheet.html

You can also use `man` ("manual"). This opens the **manual page** for another
command. Here you will find usage information, including what flags a command
accepts. 

```
$ man ls
LS(1)                                            General Commands Manual                                            LS(1)

NAME
     ls – list directory contents

SYNOPSIS
     ls [-@ABCFGHILOPRSTUWabcdefghiklmnopqrstuvwxy1%,] [--color=when] [-D format] [file ...]

DESCRIPTION
     For each operand that names a file of a type other than directory, ls displays its name as well as any requested,
     associated information.  For each operand that names a file of type directory, ls displays the names of files
     contained within that directory, as well as any requested, associated information.

     If no operands are given, the contents of the current directory are displayed.  If more than one operand is given,
     non-directory operands are displayed first; directory and non-directory operands are sorted separately and in
     lexicographical order.

     The following options are available:

<...>
```

A shortened version of this output is often available by flagging a command
with `--help`. Sometimes, too, you will need to know the version of the command
you're using. You can find this information with `--version`, or `--v`.


## Directory Structures

Remember that a CLI is just another mode of interaction with your computer.
This section will solidify that idea by showing you how to navigate your
computer's **directory structure**, the arrangement and organization of its
data.


### Paths

A directory structure is like a map of all the locations you can navigate on
your computer. These locations are **directories** (or folders), and within
them are **files** (chunks of data). Each file has an **address** on this map,
and there is a **path** that leads to it. The windowing systems of modern
computers manage these paths automatically when you click around on your
computer, but it's also possible to navigate on the command line. You'll just
need to specify paths yourself.

In a Unix environment, we do this with the following syntax:

```
/this/is/a/path/to/your/file.txt
```

This is called a **file path**. It threads through multiple directories to
point at your desired file, `file.txt`.

````{note}
Non-Unix environments, like Windows DOS, use `\` instead of `/`:

```
C:\this\is\a\path\to\your\file.txt
```

This is why Windows users had to download Git Bash. It emulates Unix-style
functionality on those machines. Generally speaking, the Unix command line is
far more pervasive than DOS, so you'll find yourself using Unix-style
commands/syntax more frequently.
````


### Path hierarchies

A directory structure is **hierarchical**. Directories are "above" or "below"
one another, and it's crucial to get a sense of this so that you can navigate
your computer. In the file path above, every `/` demarcates a new one of these
directory levels.

Use `pwd` ("print working directory") to display your current location using
the same format:

```
$ pwd
/Users/tyler/2024_dtl_llm-interpretability
```

Note that this path begins with `/`. This is the top-most directory in your
computer, called the **root**. It's like the trunk of a tree: every directory
branches off from it. But, to add some complications, directories can also
branch off one another. Whenever you make a new directory, you've created
another branch in this tree, and this branch is at the same time a branch of
root and a branch of whatever directory you're currently in.

For example, here is the structure of this reader's `data` directory:

```
data
├── bert_blurb_classifier
│   └── final
├── dickinson_poetry-foundation-poems
│   └── poems
├── james_corpus
│   ├── chapterize
│   │   └── chapterize
│   ├── chapterized
│   │   ├── 1871-watch-and-ward
│   │   ├── 1875-roderick-hudson
│   │   ├── 1877-the-american
│   │   ├── 1878-the-europeans
│   │   ├── 1879-confidence
│   │   ├── 1880-washington-square
│   │   ├── 1881-portrait-of-a-lady
│   │   ├── 1886-bostonians
│   │   ├── 1886-princess-casamassima
│   │   ├── 1888-reverberator
│   │   ├── 1890-tragic-muse
│   │   ├── 1897-spoils-poynton
│   │   ├── 1897-what-maisie-knew
│   │   ├── 1899-awkward-age
│   │   ├── 1901-sacred-found
│   │   ├── 1902-wings-of-the-dove
│   │   ├── 1903-ambassadors
│   │   ├── 1904-golden-bowl
│   │   ├── 1911-outcry
│   │   └── 1917-ivory-tower
│   └── raw
├── nyt_obituaries
│   └── texts
└── saussure
```

See all the branching paths?

```{note}
DOS again diverges from Unix in its representation of the root. For the former,
the root is usually called `C:` or `D:`, which designates the actual device on
which your data is stored.
```


### Absolute vs. relative paths

No matter the operating system, there are two different ways to specify a file
path on the command line. Recall our `pwd` output above:

```
$ pwd
/Users/tyler/2024_dtl_llm-interpretability
```

It starts with root `/`. When you see a path like this, it's showing you the
full, or **absolute**, location of a directory or file. By typing out this
path, you can navigate directly to this location, regardless of where you are
on your computer.

By contrast, a **relative** path is context-specific. It depends on where you
are in your computer's directory structure. Unix uses shorthand to denote this
location:

+ `.` denotes the current location in your computer
+ `..` denotes the directory above that location

Remember seeing these earlier with `ls -alh`? Your computer tracks locations by
placing these two symbols in every directory you create. Thus, this enables you
to use this shorthand notation to avoid typing out the entire path to a file.
This is useful if you're a ways off from root, or if, for a coding project, you
are using files that rely on a specific directory structure, which could be
ported to someone else's computer.


### Navigating with a CLI

Here's an example of absolute and relative paths. Say you are here:

```
/here/is/where/you/are/located/in/your/computer
                                       ^^^^^^^^
```

...and you want to get to `located`:

```
/here/is/where/you/are/located/in/your/computer
                       ^^^^^^^
```

You could use an absolute path. Use `cd` ("change directory") to do so:

```
$ cd /here/is/where/you/are/located
```

The logic of this path runs like this:

```
root
└── here
│   └── is
│       └── where
│           └── you
│               └── are
└──────────────────>└── located
                        └── in
                            └── your
                                └── computer
```

Alternatively, you could use a relative path:

```
$ cd ../../../
```

In contrast with the absolute path, the logic of this relative one runs like
this:

```
root
└── here
    └── is
        └── where
            └── you
                └── are
                    └── located
                    ^   └── in
                    │       └── your
                    │           └── computer
                    └───────────────┘
```

Your relative path would take you three directories up from `computer` to
`located`. That's considerably shorter than writing the path out in full, but
there's also a downside: using multiple `..` symbols makes it hard to know what
the context of the path is, and it's easy to get confused. So, there's a
trade-off between absolute and relative paths, which you'll often find yourself
making.

Finally, we can move data and directories around on our computers using paths.
To move a file, use `mv` ("move"), which works like so:

```
$ mv <location/of/file> <location/where/you/want/to/move/the/file>
```

If we're in a directory that looks like this:

```
$ ls
file.txt  subdirectory
```

...and we'd like to move `file.txt` into `subdirectory`, we use:

```
$ mv file.txt subdirectory
```

Or, maybe we want to make a new directory inside `subdirectory`. Use `mkdir`
("make directory") to make a new directory, `new`, which will sit under
`subdirectory`.

```
$ mkdir subdirectory/new
```

Now, from our current location, we move `file.txt` into `new`.

```
$ mv subdirectory/file.txt subdirectory/new/file.txt
```

This is what the directory structure looks like now:

```
current_location
└── subdirectory
    └── new
        └── file.txt
```


## Environment Setup



### Micromamba



### Jupyter Lab



## The Python Console



## Packages



### Installing packages



### Modules
