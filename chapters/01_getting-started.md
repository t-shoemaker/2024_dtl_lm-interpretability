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

:::{tip}
In the following sections, we will display CLI commands by prepending them with
an indicator character, `$`. You do not need to type this character yourself,
it's just to signal that a cell shows a command.
:::


### Input and interactions

Time to enter some commands. This is simply a matter of typing them in after
the prompt and pressing `Return`/`Enter`.

For example, `echo` will print back to screen any text that you supply the
command:

```sh
$ echo "Hello world!"
```
```
Hello world!
```

You can see what files and folders are in your current location with `ls`
("list"):

```sh
$ ls
```
```
_build      _config.yml _toc.yml    chapters    data        img         src
```

Send `ls` the name of a subfolder to show its contents:

```sh
$ ls chapters
```
```
01_getting-started.md      04_ngram-models.md         07_intro-to-llms.md        index.md
02_python-basics.md        05_vectorization.md        08_bert.md
03_data-analysis-basics.md 06_vector-spaces.md        09_gpt.md
```

Want more information? Modify the base functionality of `ls`---or any
command---with **flags**. We do this by adding a dash `-` and then a letter, or
a combination of letters, directly after the dash. Below, we send `ls` flags to
have it show all contents (`a`) with long-formatted output (`l`) in a
human-readable fashion (`h`):

```sh
$ ls -alh
```
```
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

```sh
$ <command> <optional flags> <file/data on which to run the command>
```

One caveat: since spaces are a meaningful part of the command-line syntax, file
names with spaces in them can cause problems. You'll need to **escape** these
names with `\`.

This will throw an error:

```sh
$ <command> file name.txt
```

But this will not:

```sh
$ <command> file\ name.txt
```

Be sure to read error messages when you see them. While the CLI is often quite
mute, it will do its best to show you what you did wrong when an error arises.
Below, for example, the shell explains that it cannot find a command, which
actually stems from a typing error:

```sh
$ lschapters
```
```
zsh: command not found: lschapters
```

Likewise, here, `ls` cannot find the requested folder because it does not
exist:

```sh
$ ls storage
```
```
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

```sh
$ man ls
```
```
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

```sh
/this/is/a/path/to/your/file.txt
```

This is called a **file path**. It threads through multiple directories to
point at your desired file, `file.txt`.

:::{note}
Non-Unix environments, like Windows DOS, use `\` instead of `/`:

```sh
C:\this\is\a\path\to\your\file.txt
```

This is why Windows users had to download Git Bash. It emulates Unix-style
functionality on those machines. Generally speaking, the Unix command line is
far more pervasive than DOS, so you'll find yourself using Unix-style
commands/syntax more frequently.
:::


### Path hierarchies

A directory structure is **hierarchical**. Directories are "above" or "below"
one another, and it's crucial to get a sense of this so that you can navigate
your computer. In the file path above, every `/` demarcates a new one of these
directory levels.

Use `pwd` ("print working directory") to display your current location using
the same format:

```sh
$ pwd
```
```
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

:::{note}
DOS again diverges from Unix in its representation of the root. For the former,
the root is usually called `C:` or `D:`, which designates the actual device on
which your data is stored.
:::


### Absolute vs. relative paths

No matter the operating system, there are two different ways to specify a file
path on the command line. Recall our `pwd` output above:

```sh
$ pwd
```
```
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

Besides `.` and `..`, there is also `~`, which we saw in the CLI prompt
earlier. This denotes your **home directory**, which your computer uses to
store data and configurations that are specific to you. Use `cd` ("change
directory") in combination with `~` to return to home, no matter where you are
in your computer's directory structure.

```sh
$ cd ~
```

Take a look at your prompt: it should list out `~` in the location position.


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

You could use an absolute path:

```sh
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

```sh
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

```sh
$ mv <location/of/file> <location/where/you/want/to/move/the/file>
```

If we're in a directory that looks like this:

```sh
$ ls
```
```
file.txt  subdirectory
```

...and we'd like to move `file.txt` into `subdirectory`, we use:

```sh
$ mv file.txt subdirectory
```

Or, maybe we want to make a new directory inside `subdirectory`. Use `mkdir`
("make directory") to make a new directory, `new`, which will sit under
`subdirectory`.

```sh
$ mkdir subdirectory/new
```

Now, from our current location, we move `file.txt` into `new`.

```sh
$ mv subdirectory/file.txt subdirectory/new/file.txt
```

This is what the directory structure looks like now:

```sh
current_location
└── subdirectory
    └── new
        └── file.txt
```


## Environment Setup

Processes---applications, background tasks, code, etc.---run on your computer
in a **computer environment**. These environments are collections of hardware,
software, and various configurations, and the latter two are portable between
computers (ideally). This portability is great for when you get a new computer
and want to recreate your current environment, whether by replicating certain
settings or installing external software; but the real power of porting
environments stems from the ability to freeze them and share them with others.

You'll often have to do this when writing code, because some people may not
have certain pieces of software that you have on your computer. And, as we'll
discuss a little later on, writing code often relies on specific versions of
programming language packages, which can conflict with one another---and cause
massive headaches. Using an **environment manager** to encapsulate your setup
will help you circumvent such problems and let others run your code as you
intended.


### Micromamba

We will use [`micromamba`][micro] to manage environments for the work that lies
ahead. It allows us to create new **virtual environments** and install software
into them, including different versions of Python and associated packages.
`micromamba` is based on [`mamba`][mamba], which is in turn based on
[`conda`][conda]; the three are drop-in replacements for each another, so when
you see discussions about one, know that the topic will likely port over to the
others.

[micro]: https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html
[mamba]: https://mamba.readthedocs.io/en/latest/user_guide/mamba.html
[conda]: https://docs.conda.io/en/latest/

The `micromamba` documentation (linked above) will always feature the most
recent version of installation instructions. As of this writing (summer 2024),
installing via the script option in a CLI works like so:

```sh
$ "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Running this command will download and install a script. During installation,
you'll be asked a series of questions:

+ `Micromamba binary folder? [~/.local/bin]`: Where do you want to install the
  program? Select the default by pressing `Enter`/`Return`
+ `Init shell (bash)? [Y/n]`: Configure your shell for `micromamba`? Enter `Y`
+ `Configure conda-forge? [Y/n]`: Default to the `conda-forge` repository when
  searching for packages to install? Enter `Y`
+ `Prefix location? [~/micromamba]`: Where would you like to install
  environments? The default is under your home directory, which is fine. Press
  `Enter`/`Return`

With these options set, reload your shell to initialize `micromamba`:

::::{tab-set}
:::{tab-item} zsh
```sh
$ source ~/.zshrc
```
:::
:::{tab-item} bash
```sh
$ source ~/.bashrc
```
:::
::::

:::{tip}
You don't need to run `source` to initialize `micromamba` every time you open
your CLI. Your CLI actually runs this command when on start up---we just have
the program running already.
:::

Now check the version of `micromamba`:

```sh
$ micromamba --version
```
```
1.5.8
```

If you see output like the above, you're set!

Time to create a new environment. Generally, it's a good idea to make a new
environment for every project. Below, we make one called `practice`.

```sh
$ micromamba create --name practice
```
```
Empty environment created at prefix: <path/to/practice>
```

Use the `env list` subcommand to list out all environments:

```sh
$ micromamba env list
```
```
  Name      Active  Path
──────────────────────────────────────────────────────────
  practice          <path/to/micromamba/envs/practice>
```

Activate an environment with the `activate <enviroment>` subcommand.

```sh
$ micromamba activate practice
```

Note your prompt. It will update to reflect that you're in a `micromamba`
environment:

```sh
(practice) you@your_computer ~$
```

Let's install some software. Below, we install [`ripgrep`][rg], which enables
you to search text files in a directory:

```sh
$ micromamba install ripgrep
```

[rg]: https://github.com/BurntSushi/ripgrep

When prompted, enter `Y` to confirm that you'd like to install the program.

Now, use `ripgrep` to search for "NLP" in `chapters`:

```sh
$ rg -i "NLP" chapters
```
```
chapters/05_vectorization.md
648:  in the same direction as the first. Most vector operations in NLP use the

chapters/02_python-basics.md
309:all repeated tokens. The result will be a set of in NLP are called **types**:

chapters/01_getting-started.md
664:Now, use `ripgrep` to search for "NLP" in `chapters`:
667:$ rg -i "NLP" chapters
```

See how it picked up the very text you're reading now?

Use the `deactivate` subcommand to deactivate an environment.

```sh
$ micromamba deactivate
```

Now, if you try to use `ripgrep`, you'll get an error:

```sh
$ rg -i "NLP" chapters
```
```
zsh: command not found: rg
```

Why the error? You've installed `ripgrep` into a separate environment managed
by `micromamba`, which is sealed off from the rest of your computer. 

To see all packages installed in an environment, use `list`:

```sh
$ micromamba activate practice
$ micromamba list
```
```
List of packages in environment: "<path/to/micromamba/envs/practice>"

  Name           Version  Build        Channel
────────────────────────────────────────────────────
  _libgcc_mutex  0.1      conda_forge  conda-forge
  _openmp_mutex  4.5      2_gnu        conda-forge
  libgcc-ng      13.2.0   h77fa898_7   conda-forge
  libgomp        13.2.0   h77fa898_7   conda-forge
  ripgrep        14.1.0   he8a937b_0   conda-forge
```

Export your environment with the `env export` subcommand:

```sh
$ micromamba env export
```
```
name: practice
channels:
- conda-forge
dependencies:
- _libgcc_mutex=0.1=conda_forge
- _openmp_mutex=4.5=2_gnu
- libgcc-ng=13.2.0=h77fa898_7
- libgomp=13.2.0=h77fa898_7
- ripgrep=14.1.0=he8a937b_0
```

Those letter and number combinations after each package are specific versions,
which another person could use to recreate your current environment. That said,
sometimes operating system differences complicate versioning, so set the
`--from-history` flag to get the package names only:

```sh
$ micromamba env export --from-history
name: practice
channels:
- conda-forge
dependencies:
- ripgrep
```

:::{note}
If you installed a package with a specific version, e.g.:

```sh
$ micromamba install ripgrep=12.1
```

...the `--from-history` flag will preserve this:

```sh
$ micromamba env export --from-history
```
```
name: practice
channels:
- conda-forge
dependencies:
- ripgrep=12.1
```
:::

Redirect the output of your environment export to a YAML file:

```sh
$ micromamba env export --from-history > practice.yml
```

And that way someone can install it on their own computer, using:

```sh
$ micromamba env create --file practice.yml
```

Finally, to remove an environment, deactivate it and run the following:

```sh
$ micromamba env remove --name practice
```


### Jupyter Lab



## The Python Console



## Packages



### Installing packages



### Modules
