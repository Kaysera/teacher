"""
The :mod:`teacher.datasets` module includes the different databases
used to run experiments with the :mod:`teacher` package.

Available datasets
-------------------

This module includes *load* methods for the following datasets that are included:

.. _adult-dataset:

* Adult: `Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a \
Decision-Tree Hybrid", Proceedings of the Second International Conference on  \
Knowledge Discovery and Data Mining, 1996 <https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf>`_ \
`[dataset] <https://archive.ics.uci.edu/ml/datasets/adult>`_

.. _beer-dataset:

* Beer: `G. Castellano, C. Castiello, and A. M. Fanelli, “The FISDeT software: \
Application to beer style classification,” in Proc. IEEE Int. Conf. Fuzzy \
Syst., 2017, pp. 1–6. <https://doi.org/10.1109/FUZZ-IEEE.2017.8015503>`_ \

.. _breast-dataset:

* Breast: `O. L. Mangasarian and W. H. Wolberg, “Cancer diagnosis via linear \
programming,” Dept. Comput. Sci., Univ. Wisconsin-Madison, Madison, \
WI, USA, Tech. Rep., 1990 <https://doi.org/10.1287/opre.43.4.570>`_. \
`[dataset] <https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29>`_

.. _compas-dataset:

* Compas:  `J. Skeem and J. Eno Louden, “Assessment of evidence on the quality of \
the correctional offender management profiling for alternative sanctions \
(COMPAS),” California Dept. Corrections Rehabilitation, 2007. \
<https://cpb-us-e2.wpmucdn.com/sites.uci.edu/dist/0/1149/files/2013/06/CDCR\
    -Skeem-EnoLouden-COMPASeval-SECONDREVISION-final-Dec-28-07.pdf>`_ \
`[dataset] <https://www.kaggle.com/datasets/danofer/compass>`_

.. _german-dataset:

* German: `[dataset] <https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)>`_

.. _heloc-dataset:

* Heloc: `[dataset] <https://community.fico.com/s/explainable-machinelearning-challenge>`_

.. _pima-dataset:

* Pima: `[dataset] <https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database>`_


Dataset format
---------------

The different methods return a `dict` with the following keys:

    * name : :class:`str`, Name of the dataset

    * df : :class:`pandas.DataFrame` Pandas DataFrame with the original data

    * columns : :class:`list`, columns of the DataFrame

    * class_name : :class:`str`, name of the class variable

    * possible_outcomes : :class:`list`, the values of the class column

    * type_features : :class:`dict`, the variables grouped by type

    * features_type : :class:`dict`, the type of each feature

    * discrete : :class:`list`, columns to be considered to have discrete values

    * continuous : :class:`list`, columns to be considered to have continuous values

    * idx_features : :class:`dict`, column name of each column once arranged in a NumPy array

    * label_encoder : :class:`sklearn.preprocessing.LabelEncoder`, label encoder for the discrete values

    * X : :class:`numpy.ndarray`, all columns except for the class

    * y : :class:`numpy.ndarray`, class column

Functions
----------
:meth:`load_adult`
    Loads the `adult <adult-dataset_>`_ dataset

:meth:`load_beer`
    Loads the `beer <beer-dataset_>`_ dataset

:meth:`load_breast`
    Loads the `breast <breast-dataset_>`_ dataset

:meth:`load_compas`
    Loads the `compas <compas-dataset_>`_ dataset

:meth:`load_heloc`
    Loads the `heloc <heloc-dataset_>`_ dataset

:meth:`load_pima`
    Loads the `pima <pima-dataset_>`_ dataset


------------------------
"""

# =============================================================================
# Imports
# =============================================================================

# Local application
from ._base import (load_german,
                    load_adult,
                    load_compas,
                    load_heloc,
                    load_beer,
                    load_pima,
                    load_breast,
                    load_basket,
                    load_phishing,
                    load_flavia,
                    load_iris,
                    load_wine)


# =============================================================================
# Public objects
# =============================================================================

# Set the classes that are accessible
# from the module teacher.datasets
__all__ = [
    "load_adult",
    "load_basket",
    "load_beer",
    "load_breast",
    "load_compas",
    "load_german",
    "load_heloc",
    "load_pima",
    "load_phishing",
    "load_flavia",
    "load_iris",
    "load_wine"
]
