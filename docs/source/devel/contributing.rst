.. _contributing:

============
Contributing
============

This project is a community effort, and everyone is welcome to
contribute. Everyone within the community
is expected to abide by our
`code of conduct <https://github.com/Kaysera/teacher/blob/main/.github/CODE_OF_CONDUCT.md>`_.

The project is hosted on
https://github.com/Kaysera/teacher

.. _new_contributors:

Issues for new contributors
---------------------------

While any contributions are welcome, we have marked some issues as
particularly suited for new contributors by the label
`good first issue <https://github.com/Kaysera/teacher/labels/good%20first%20issue>`_
These are well documented issues, that do not require a deep understanding of
the internals of Teacher. The issues may additionally be tagged with a
difficulty. ``Difficulty: Easy`` is suited for people with little Python experience.
``Difficulty: Medium`` and ``Difficulty: Hard`` are not trivial to solve and
require more thought and programming experience.

In general, the Teacher project does not assign issues. Issues are
"assigned" or "claimed" by opening a PR; there is no other assignment
mechanism. If you have opened such a PR, please comment on the issue thread to
avoid duplication of work. Please check if there is an existing PR for the
issue you are addressing. If there is, try to work with the author by
submitting reviews of their code or commenting on the PR rather than opening
a new PR; duplicate PRs are subject to being closed.  However, if the existing
PR is an outline, unlikely to work, or stalled, and the original author is
unresponsive, feel free to open a new PR referencing the old one.

.. _submitting-a-bug-report:

Submitting a bug report
=======================

If you find a bug in the code or documentation, do not hesitate to submit a
ticket to the
`Issue Tracker <https://github.com/Kaysera/teacher/issues>`_. You are
also welcome to post feature requests or pull requests.

If you are reporting a bug, please do your best to include the following:

1. A short, top-level summary of the bug. In most cases, this should be 1-2
   sentences.

2. A short, self-contained code snippet to reproduce the bug, ideally allowing
   a simple copy and paste to reproduce. Please do your best to reduce the code
   snippet to the minimum required.

3. The actual outcome of the code snippet.

4. The expected outcome of the code snippet.

5. The Teacher version, Python version and platform that you are using. You
   can grab the version with the following commands::

      >>> import teacher
      >>> teacher.__version__
      '0.1b2'
      >>> import platform
      >>> platform.python_version()
      '3.9.0'

We have preloaded the issue creation page with a Markdown form that you can
use to organize this information.

Thank you for your help in keeping bug reports complete, targeted and descriptive.

.. _request-a-new-feature:

Requesting a new feature
========================

Please post feature requests to the
`Issue Tracker <https://github.com/Kaysera/teacher/issues>`_.

The Teacher developers will give feedback on the feature proposal. Since
Teacher is an open source project with limited resources, we encourage
users to then also
:ref:`participate in the implementation <contributing-code>`.

.. _contributing-code:

Contributing code
=================

.. _how-to-contribute:

How to contribute
-----------------

The preferred way to contribute to Teacher is to fork the `main
repository <https://github.com/Kaysera/teacher/>`__ on GitHub,
then submit a "pull request" (PR).

A brief overview is:

1. `Create an account <https://github.com/join>`_ on GitHub if you do not
   already have one.

2. Fork the `project repository <https://github.com/Kaysera/teacher>`_:
   click on the 'Fork' button near the top of the page. This creates a copy of
   the code under your account on the GitHub server.

3. Clone this copy to your local disk::

      git clone https://github.com/<YOUR GITHUB USERNAME>/teacher.git

4. Enter the directory and install the local version of Teacher.
   See :ref:`installing_for_devs` for instructions

5. Create a branch to hold your changes::

      git checkout -b my-feature origin/main

   and start making changes. Never work in the ``main`` branch!

6. Work on this copy, on your computer, using Git to do the version control.
   When you're done editing e.g., ``fuzzy/fuzzy_set.py``, do::

      git add fuzzy/fuzzy_set.py
      git commit

   to record your changes in Git, then push them to GitHub with::

      git push -u origin my-feature

Finally, go to the web page of your fork of the Teacher repo, and click
'Pull request' to send your changes to the maintainers for review.

.. seealso::

  * `Git documentation <https://git-scm.com/doc>`_
  * `Git-Contributing to a Project <https://git-scm.com/book/en/v2/GitHub-Contributing-to-a-Project>`_
  * `Introduction to GitHub  <https://lab.github.com/githubtraining/introduction-to-github>`_

Contributing pull requests
--------------------------

It is recommended to check that your contribution complies with the following
rules before submitting a pull request:

* If your pull request addresses an issue, please use the title to describe the
  issue and mention the issue number in the pull request description to ensure
  that a link is created to the original issue.

* All public methods should have informative docstrings with sample usage when
  appropriate. Use the `numpy docstring standard
  <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

* Formatting should follow the recommendations of PEP8_, as enforced by
  flake8_.  You can check flake8 compliance from the command line with ::

    python -m pip install flake8
    flake8 /path/to/module.py

  or your editor may provide integration with it.

  .. _PEP8: https://www.python.org/dev/peps/pep-0008/
  .. _flake8: https://flake8.pycqa.org/

* Changes (both new features and bugfixes) should have good test coverage. See
  :ref:`testing` for more details.

* Import the following modules using the standard scipy conventions::

     import numpy as np
     import numpy.ma as ma

.. note::

    The current state of the Teacher code base is not compliant with all
    of those guidelines, but we expect that enforcing those constraints on all
    new contributions will move the overall code base quality in the right
    direction.


.. seealso::

  * :ref:`testing`
  * :ref:`documenting-teacher`


.. _contributing_documentation:

Contributing documentation
==========================

You as an end-user of Teacher can make a valuable contribution because you
more clearly see the potential for improvement than a core developer. For example, you can:

- Fix a typo
- Clarify a docstring
- Write or update a comprehensive :ref:`tutorial <tutorials>`

The documentation source files live in the same GitHub repository as the code.
Contributions are proposed and accepted through the pull request process.
For details see :ref:`how-to-contribute`.

If you have trouble getting started, you may instead open an `issue`_
describing the intended improvement.

.. _issue: https://github.com/Kaysera/teacher/issues

.. seealso::
  * :ref:`documenting-teacher`

.. _other_ways_to_contribute:

Other ways to contribute
========================

It also helps us if you spread the word: reference the project from your blog
and articles or link to it from your website!  If Teacher contributes to a
project that leads to a scientific publication, please follow the
:doc:`/users/project/citing` guidelines.

.. _coding_guidelines:
