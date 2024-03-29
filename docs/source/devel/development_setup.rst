.. _installing_for_devs:

=====================================
Setting up Teacher for development
=====================================

.. _dev-environment:

Creating a dedicated environment
================================
You should set up a dedicated environment to decouple your Teacher
development from other Python and Teacher installations on your system.
Here we use python's virtual environment `venv`_, but you may also use others
such as conda.

.. _venv: https://docs.python.org/3/library/venv.html

A new environment can be set up with ::

   python -m venv <file folder location>

and activated with one of the following::

   source <file folder location>/bin/activate  # Linux/macOS
   <file folder location>\Scripts\activate.bat  # Windows cmd.exe
   <file folder location>\Scripts\Activate.ps1  # Windows PowerShell

Whenever you plan to work on Teacher, remember to activate the development
environment in your shell.

Retrieving the latest version of the code
=========================================

Teacher is hosted at https://github.com/Kaysera/teacher.git.

You can retrieve the latest sources with the command::

    git clone https://github.com/Kaysera/teacher.git

This will place the sources in a directory :file:`teacher` below your
current working directory.

If you have the proper privileges, you can use ``git@`` instead of
``https://``, which works through the ssh protocol and might be easier to use
if you are using 2-factor authentication.

Installing Teacher in editable mode
======================================
Install Teacher in editable mode from the :file:`teacher` directory
using the command ::

    python -m pip install -ve .

The 'editable/develop mode', builds everything and places links in your Python
environment so that Python will be able to import Teacher from your
development source directory.  This allows you to import your modified version
of Teacher without re-installing after every change. 

Installing pre-commit hooks
===========================
You can optionally install `pre-commit <https://pre-commit.com/>`_ hooks.
These will automatically check flake8 and other style issues when you run
``git commit``. The hooks are defined in the top level
``.pre-commit-config.yaml`` file. To install the hooks ::

    python -m pip install pre-commit
    pre-commit install

The hooks can also be run manually. All the hooks can be run, in order as
listed in ``.pre-commit-config.yaml``, against the full codebase with ::

    pre-commit run --all-files

To run a particular hook manually, run ``pre-commit run`` with the hook id ::

    pre-commit run <hook id> --all-files
