.. _troubleshooting-faq:

.. redirect-from:: /faq/troubleshooting_faq

***************
Troubleshooting
***************

.. contents::
   :backlinks: none

.. teacher-version:

Obtaining Teacher version
============================

To find out your Teacher version number, import it and print the
``__version__`` attribute::

    >>> import teacher
    >>> teacher.__version__
    '0.1.dev0'


.. _locating-teacher-install:

:file:`teacher` install location
===================================

You can find what directory Teacher is installed in by importing it
and printing the ``__file__`` attribute::

    >>> import teacher
    >>> teacher.__file__
    '/Users/guillermo/Documents/teacher/__init__.py'

.. _reporting-problems:

Getting help
============

There are a number of good resources for getting help with Teacher.
There is a good chance your question has already been asked:

- `GitHub issues <https://github.com/Kaysera/teacher/issues>`_.


If you are unable to find an answer to your question through search, please
provide the following information in your issue in Github:

* Your operating system (Linux/Unix users: post the output of ``uname -a``).

* Teacher version::

     python -c "import teacher; print(teacher.__version__)"

* Where you obtained Teacher (e.g., GitHub or PyPI).

* If the problem is reproducible, please try to provide a *minimal*, standalone
  Python script that demonstrates the problem.  This is *the* critical step.
  If you can't post a piece of code that we can run and reproduce your error,
  the chances of getting help are significantly diminished.  Very often, the
  mere act of trying to minimize your code to the smallest bit that produces
  the error will help you find a bug in *your* code that is causing the
  problem.