.. _testing:

=======
Testing
=======

Teacher uses the pytest_ framework.

The tests are in the :file:`tests` in each module.

.. _pytest: http://doc.pytest.org/en/latest/

.. _testing_requirements:

Requirements
------------

To run the tests you will need to
:ref:`set up Teacher for development <installing_for_devs>`.


Running the tests
-----------------

In the root directory of your development repository run::

   python -m pytest


pytest can be configured via a lot of `command-line parameters`_. Some
particularly useful ones are:

=============================  ===========
``-v`` or ``--verbose``        Be more verbose
``--capture=no`` or ``-s``     Do not capture stdout
=============================  ===========

To run a single test from the command line, you can provide a file path,
optionally followed by the function separated by two colons, e.g., (tests do
not need to be installed, but Teacher should be)::

  pytest teacher/fuzzy/tests/test_base.py::test_get_dataset_membership


.. _command-line parameters: http://doc.pytest.org/en/latest/usage.html


Using GitHub Actions for CI
---------------------------

`GitHub Actions <https://docs.github.com/en/actions>`_ is a hosted CI system
"in the cloud".

GitHub Actions is configured to receive notifications of new commits to GitHub
repos and to run builds or tests when it sees these new commits. It looks for a
YAML files in ``.github/workflows`` to see how to test the project.

GitHub Actions is already enabled for the `main Teacher GitHub repository
<https://github.com/Kaysera/teacher/>`_.

GitHub Actions should be automatically enabled for your personal Teacher
fork once the YAML workflow files are in it. It generally isn't necessary to
look at these workflows, since any pull request submitted against the main
Teacher repository will be tested.

You can see the GitHub Actions results at
https://github.com/your_GitHub_user_name/teacher/actions.
