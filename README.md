n-body sample codes
===================

This git repository contains a few n-body codes written for both CPUs
(ANSI C) and GPUs (CUDA/C).  Our goal here is not to implement
state-of- the-art algorithms.  Instead, we want to present
optimization techniques for education purpose for the PIRE 2018 Winter
School.  Therefore, we keep the codes as simple as possible.  We will
iterate the codes and use git to keep track of the improvements.

You may visit

    https://github.com/bhpire/nbody

to browse the repository on-line.  Nevertheless, we strongly recommend
that you install `CUDA`, clone the codes by `git`:

    git clone https://github.com/bhpire/nbody.git

and try the codes yourself.

to download the latest version of this code.  You can then use `gitk`
or `tig` to look at the revision history and study the optimization
techniques that we use.  You can also use, for exampmle, `git difftool
HEAD^ nbody.c` to look at the side-by-side diff of "nbody.c".
