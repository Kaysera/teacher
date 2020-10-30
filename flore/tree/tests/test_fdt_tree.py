import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pandas as pd

from flore.fuzzy import get_fuzzy_points_entropy, get_fuzzy_triangle
from flore.tree import FDT


def test_tree():
    
    theory = np.array([0,0,1,1,1,2,2,2,3,3,3,6,7,9,9,9,10])
    practice = np.array([0,0,0,0,0,1,1,2,5,6,8,9,10,10,10])

    df = pd.DataFrame(([i, j, i+j >= 5] for i, j in zip(theory, practice)), columns=['theory', 'practice', 'class'])

    df_numerical_columns = ['theory', 'practice']
    class_name = 'class'

    X = df[df_numerical_columns]
    y = df[class_name]

    fuzzy_points = get_fuzzy_points_entropy(df, df_numerical_columns, class_name)


    # THIS IS GET_FUZZY_SET_DATAFRAME
    fuzzy_set_dataframe = {}
    for column in df_numerical_columns:
        labels = [f'{label}' for label in fuzzy_points[column]]
        fuzzy_set_dataframe[column] = get_fuzzy_triangle(df[column].to_numpy(), list(zip(labels,fuzzy_points[column])), False)
    # PARAMETRIZE FOR IT TO NOT NEED LABELS' NAMES
    print('\n--------------------------------')
    print('Begin tree')
    print('--------------------------------')
    fdt = FDT(df_numerical_columns, fuzzy_set_dataframe)
    fdt.fit(X,y)

    print(fdt.tree)



    assert 1 == 2-1