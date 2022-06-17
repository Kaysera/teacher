.. _documenting-teacher:

=====================
Writing documentation
=====================

Getting started
===============

General file structure
----------------------

All documentation is built from the :file:`docs/source`.  The :file:`docs/source`
directory contains configuration files for Sphinx and reStructuredText
(ReST_; ``.rst``) files that are rendered to documentation pages.

Documentation is created in three ways.  First, API documentation
(:file:`docs/source/api`) is created by Sphinx_ from
the docstrings of the classes in the Teacher library. All :file:`docs/source/api` are created
when the documentation is built.  See :ref:`writing-docstrings` below.

Second, Teacher has narrative docs written in ReST_ in subdirectories of
:file:`docs/source/users/`.  If you would like to add new documentation that is suited
to an ``.rst`` file rather than a gallery or tutorial example, choose an
appropriate subdirectory to put it in, and add the file to the table of
contents of :file:`index.rst` of the subdirectory.  See
:ref:`writing-rest-pages` below.

.. note::

  Don't directly edit the ``.rst`` files in :file:`docs/source/api`.
  Sphinx_ regenerates these files in these directories when building documentation.

Setting up the doc build
------------------------

The documentation for Teacher is generated from reStructuredText (ReST_)
using the Sphinx_ documentation generation tool.

To build the documentation you will need to
:ref:`set up Teacher for development <installing_for_devs>`. 

Building the docs
-----------------

The documentation sources are found in the :file:`docs/source/` directory in the trunk.
The configuration file for Sphinx is :file:`docs/source/conf.py`. It controls which
directories Sphinx parses, how the docs are built, and how the extensions are
used. To build the documentation in html format, cd into :file:`docs/` and run:

.. code-block:: sh

   make html

Other useful invocations include

.. code-block:: sh

   # Delete built files.  May help if you get errors about missing paths or
   # broken links.
   make clean

   # Build pdf docs.
   make latexpdf


Showing locally built docs
--------------------------

The built docs are available in the folder :file:`docs/build/html`. 

.. _writing-rest-pages:

Writing ReST pages
==================

Most documentation is either in the docstrings of individual
classes and methods, in explicit ``.rst`` files, or in examples and tutorials.
All of these use the ReST_ syntax and are processed by Sphinx_.

The `Sphinx reStructuredText Primer
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_ is
a good introduction into using ReST. More complete information is available in
the `reStructuredText reference documentation
<https://docutils.sourceforge.io/rst.html#reference-documentation>`_.

This section contains additional information and conventions how ReST is used
in the Teacher documentation.

Formatting and style conventions
--------------------------------

It is useful to strive for consistency in the Teacher documentation.  Here
are some formatting and style conventions that are used.

Section formatting
~~~~~~~~~~~~~~~~~~

For everything but top-level chapters,  use ``Upper lower`` for
section titles, e.g., ``Possible hangups`` rather than ``Possible
Hangups``

We aim to follow the recommendations from the
`Python documentation <https://devguide.python.org/documenting/#sections>`_
and the `Sphinx reStructuredText documentation <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#sections>`_
for section markup characters, i.e.:

- ``#`` with overline, for parts. This is reserved for the main title in
  ``index.rst``. All other pages should start with "chapter" or lower.
- ``*`` with overline, for chapters
- ``=``, for sections
- ``-``, for subsections
- ``^``, for subsubsections
- ``"``, for paragraphs

This may not yet be applied consistently in existing docs. Please open
:ref:`an issue <contributing_documentation>` to notify any inconsistencies. 

Function arguments
~~~~~~~~~~~~~~~~~~

Function arguments and keywords within docstrings should be referred to using
the ``*emphasis*`` role. This will keep Teacher's documentation consistent
with Python's documentation:

.. code-block:: rst

  Here is a description of *argument*

Do not use the ```default role```:

.. code-block:: rst

   Do not describe `argument` like this.  As per the next section,
   this syntax will (unsuccessfully) attempt to resolve the argument as a
   link to a class or method in the library.

nor the ````literal```` role:

.. code-block:: rst

   Do not describe ``argument`` like this.


.. _internal-section-refs:

Referring to other documents and sections
-----------------------------------------

Sphinx_ allows internal references_ between documents.

Documents can be linked with the ``:doc:`` directive:

.. code-block:: rst

   See the :doc:`/users/installing/index`

will render as:

  See the :doc:`/users/installing/index`

Sections can also be given reference names.  For instance from the
:doc:`/users/installing/index` link:

.. code-block:: rst

  .. _install_from_source:

  ======================
  Installing from source
  ======================

  If you are interested in contributing to Teacher development,
  running the latest source code, or just like to build everything
  yourself, it is not difficult to build Teacher from source.

and refer to it using the standard reference syntax:

.. code-block:: rst

   See :ref:`install_from_source`

will give the following link: :ref:`install_from_source`

To maximize internal consistency in section labeling and references,
use hyphen separated, descriptive labels for section references.
Keep in mind that contents may be reorganized later, so
avoid top level names in references like ``user`` or ``devel``
unless necessary

In addition, since underscores are widely used by Sphinx itself, use
hyphens to separate words.

.. _referring-to-other-code:

Referring to other code
-----------------------

To link to other methods, classes, or modules in Teacher you can use
back ticks, for example:

.. code-block:: rst

  `teacher.fuzzy.FuzzySet`

generates a link like this: `teacher.fuzzy.FuzzySet`.

*Note:* We use the sphinx setting ``default_role = 'obj'`` so that you don't
have to use qualifiers like ``:class:``, ``:func:``, ``:meth:`` and the likes.

Often, you don't want to show the full package and module name. As long as the
target is unambiguous you can simply leave them out:

.. code-block:: rst

  `.FuzzySet`

and the link still works: `.FuzzySet`.

Other packages can also be linked via
`intersphinx <http://www.sphinx-doc.org/en/master/ext/intersphinx.html>`_:

.. code-block:: rst

  `numpy.mean`

will return this link: `numpy.mean`.  This works for Python, Numpy, Scipy,
and Pandas (full list is in :file:`doc/conf.py`).  If external linking fails,
you can check the full list of referenceable objects with the following
commands::

  python -m sphinx.ext.intersphinx 'https://docs.python.org/3/objects.inv'
  python -m sphinx.ext.intersphinx 'https://numpy.org/doc/stable/objects.inv'
  python -m sphinx.ext.intersphinx 'https://docs.scipy.org/doc/scipy/objects.inv'
  python -m sphinx.ext.intersphinx 'https://pandas.pydata.org/pandas-docs/stable/objects.inv'

.. _rst-figures-and-includes:

Including files
-----------------

Files can be included verbatim.  For instance the ``LICENSE`` file is included
at :ref:`license-agreement` using ::

  .. literalinclude:: ../../../../LICENSE

.. _writing-docstrings:

Writing docstrings
==================

Most of the API documentation is written in docstrings. These are comment
blocks in source code that explain how the code works.

.. note::

   Some parts of the documentation do not yet conform to the current
   documentation style. If in doubt, follow the rules given here and not what
   you may see in the source code. Pull requests updating docstrings to
   the current style are very welcome.

All new or edited docstrings should conform to the `numpydoc docstring guide`_.
Much of the ReST_ syntax discussed above (:ref:`writing-rest-pages`) can be
used for links and references.  These docstrings eventually populate the
:file:`docs/source/api` directory and form the reference documentation for the
library.

Example docstring
-----------------

An example docstring looks like:

.. code-block:: python

  def generate_dataset(df, columns, class_name, discrete, name):
      """Generate the dataset suitable for LORE usage

      Parameters
      ----------
      df : pandas.core.frame.DataFrame
          Pandas DataFrame with the original data to prepare
      columns : list
          List of the columns used in the dataset
      class_name : str
          Name of the class column
      discrete : list
          List with all the columns to be considered to have discrete values
      name : str
          Name of the dataset

      Returns
      -------
      dataset : dict
          Dataset as a dictionary with the following elements:
              name : Name of the dataset
              df : Pandas DataFrame with the original data
              columns : list of the columns of the DataFrame
              class_name : name of the class variable
              possible_outcomes : list with all the values of the class column
              type_features : dict with all the variables grouped by type
              features_type : dict with the type of each feature
              discrete : list with all the columns to be considered to have discrete values
              continuous : list with all the columns to be considered to have continuous values
              idx_features : dict with the column name of each column once arranged in a NumPy array
              label_encoder : label encoder for the discrete values
              X : NumPy array with all the columns except for the class
              y : NumPy array with the class column
      """

See the `~.datasets` documentation for how this renders.

The Sphinx_ website also contains plenty of documentation_ concerning ReST
markup and working with Sphinx in general.

Formatting conventions
----------------------

The basic docstring conventions are covered in the `numpydoc docstring guide`_
and the Sphinx_ documentation.  Some Teacher-specific formatting conventions
to keep in mind:

Quote positions
~~~~~~~~~~~~~~~
The quotes for single line docstrings are on the same line (pydocstyle D200)::

  def _compare_rules_FID3(factual, counter_rule):
      """Compare two rules according to the `FID3` algorithm"""

The quotes for multi-line docstrings are on separate lines (pydocstyle D213)::

  def i_counterfactual(instance, rule_list, class_val, df_numerical_columns):
      """Returns a list that contains the counterfactual with respect to the instance
      for each of the different class values not predicted, as explained in [ref]

      [...]
      """

Function arguments
~~~~~~~~~~~~~~~~~~
Function arguments and keywords within docstrings should be referred to
using the ``*emphasis*`` role. This will keep Teacher's documentation
consistent with Python's documentation:

.. code-block:: rst

  If *linestyles* is *None*, the default is 'solid'.

Do not use the ```default role``` or the ````literal```` role:

.. code-block:: rst

  Neither `argument` nor ``argument`` should be used.


Quotes for strings
~~~~~~~~~~~~~~~~~~
Teacher does not have a convention whether to use single-quotes or
double-quotes.  There is a mixture of both in the current code.

Use simple single or double quotes when giving string values, e.g.

.. code-block:: rst

  'entropy' uses fuzzy entropy to compute the fuzzy sets.

  No ``'extra'`` literal quotes.

The use of extra literal quotes around the text is discouraged. While they
slightly improve the rendered docs, they are cumbersome to type and difficult
to read in plain-text docs.

Parameter type descriptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The main goal for parameter type descriptions is to be readable and
understandable by humans. If the possible types are too complex use a
simplification for the type description and explain the type more
precisely in the text.

Generally, the `numpydoc docstring guide`_ conventions apply. The following
rules expand on them where the numpydoc conventions are not specific.

Use ``float`` for a type that can be any number.

Use ``(float, float)`` to describe a 2D position. The parentheses should be
included to make the tuple-ness more obvious.

Use ``array-like`` for homogeneous numeric sequences, which could
typically be a numpy.array. Dimensionality may be specified using ``2D``,
``3D``, ``n-dimensional``. If you need to have variables denoting the
sizes of the dimensions, use capital letters in brackets
(``(M, N) array-like``). When referring to them in the text they are easier
read and no special formatting is needed. Use ``array`` instead of
``array-like`` for return types if the returned object is indeed a numpy array.

``float`` is the implicit default dtype for array-likes. For other dtypes
use ``array-like of int``.

Some possible uses::

  2D array-like
  (N,) array-like
  (M, N) array-like
  (M, N, 3) array-like
  array-like of int

Non-numeric homogeneous sequences are described as lists, e.g.::

  list of str
  list of `.Rule`

Referencing types
~~~~~~~~~~~~~~~~~
Generally, the rules from referring-to-other-code_ apply. More specifically:

Use full references ```~teacher.fuzzy.FuzzySet``` with an
abbreviation tilde in parameter types. While the full name helps the
reader of plain text docstrings, the HTML does not need to show the full
name as it links to it. Hence, the ``~``-shortening keeps it more readable.

Use abbreviated links ```.FuzzySet``` in the text.

.. code-block:: rst

   norm : `~teacher.fuzzy.FuzzySet`, optional
        A `.FuzzySet` is used to represent a membership function in a range of the discourse universe

Default values
~~~~~~~~~~~~~~
As opposed to the numpydoc guide, parameters need not be marked as
*optional* if they have a simple default:

- use ``{name} : {type}, default: {val}`` when possible.
- use ``{name} : {type}, optional`` and describe the default in the text if
  it cannot be explained sufficiently in the recommended manner.

The default value should provide semantic information targeted at a human
reader. In simple cases, it restates the value in the function signature.
If applicable, units should be added.

.. code-block:: rst

   Prefer:
       interval : int, default: 1000ms
   over:
       interval : int, default: 1000

If *None* is only used as a sentinel value for "parameter not specified", do
not document it as the default. Depending on the context, give the actual
default, or mark the parameter as optional if not specifying has no particular
effect.


Inheriting docstrings
---------------------

If a subclass overrides a method but does not change the semantics, we can
reuse the parent docstring for the method of the child class. Python does this
automatically, if the subclass method does not have a docstring.

Use a plain comment ``# docstring inherited`` to denote the intention to reuse
the parent docstring. That way we do not accidentally create a docstring in
the future::

    class A:
        def foo():
            """The parent docstring."""
            pass

    class B(A):
        def foo():
            # docstring inherited
            pass

.. _ReST: https://docutils.sourceforge.io/rst.html
.. _Sphinx: http://www.sphinx-doc.org
.. _documentation: https://www.sphinx-doc.org/en/master/contents.html
.. _index: http://www.sphinx-doc.org/markup/para.html#index-generating-markup
.. _`Sphinx Gallery`: https://sphinx-gallery.readthedocs.io/en/latest/
.. _references: https://www.sphinx-doc.org/en/stable/usage/restructuredtext/roles.html
.. _`numpydoc docstring guide`: https://numpydoc.readthedocs.io/en/latest/format.html