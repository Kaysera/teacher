"""
This module gathers the different datasets used to run the experiments, as well
as a function to take a new generic dataset and return it in a format understandable
by the library
"""

# =============================================================================
# Imports
# =============================================================================

# Standard
from os.path import dirname

# Third party
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# Local application
from teacher.utils import recognize_features_type, set_discrete_continuous, label_encode


# =============================================================================
# Constants
# =============================================================================

MODULE_PATH = dirname(__file__)


# =============================================================================
# Functions
# =============================================================================

def generate_dataset(df, columns, class_name, discrete, name, normalize=False):
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
    normalize : bool
        Whether to normalize the continuous features or not

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
            normalize_scaler : scaler used to normalize the continuous features
    """
    possible_outcomes = list(df[class_name].unique())

    type_features, features_type = recognize_features_type(df, class_name)
    discrete, continuous = set_discrete_continuous(columns, type_features, class_name, discrete, continuous=None)

    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    idx_features = {i: col for i, col in enumerate(columns_tmp)}
    if normalize:
        scaler = StandardScaler()
        df[continuous] = scaler.fit_transform(df[continuous])
    else:
        scaler = None
    # Dataset Preparation for Scikit Alorithms
    df_le, label_encoder = label_encode(df, discrete)
    X = df_le.loc[:, df_le.columns != class_name].values
    y = df_le[class_name].values

    dataset = {
        'name': name,
        'df': df,
        'columns': list(columns),
        'class_name': class_name,
        'possible_outcomes': possible_outcomes,
        'type_features': type_features,
        'features_type': features_type,
        'discrete': discrete,
        'continuous': continuous,
        'idx_features': idx_features,
        'label_encoder': label_encoder,
        'normalize_scaler': scaler,
        'X': X,
        'y': y,
    }

    return dataset


def load_german(normalize=False):
    """
    Load and return the german credit dataset.

    Returns
    -------
    dataset : dict
    """

    # Read Dataset
    df = pd.read_csv(MODULE_PATH + '/data/german_credit.csv', delimiter=',')

    # Features Categorization
    columns = df.columns
    class_name = 'default'

    discrete = ['installment_as_income_perc', 'present_res_since', 'credits_this_bank', 'people_under_maintenance']

    return generate_dataset(df, columns, class_name, discrete, 'german_credit', normalize)


def load_adult(normalize=False):
    """
    Load and return the adult dataset.

    Returns
    -------
    dataset : dict
    """

    # Read Dataset
    df = pd.read_csv(MODULE_PATH + '/data/adult.csv', delimiter=',', skipinitialspace=True)

    # Remove useless columns
    del df['fnlwgt']
    del df['education-num']

    # Remove Missing Values
    for col in df.columns:
        df[col].replace('?', df[col].value_counts().index[0], inplace=True)

    # Features Categorization
    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    class_name = 'class'

    discrete = []
    return generate_dataset(df, columns, class_name, discrete, 'adult', normalize)


def load_compas(normalize=False):
    """
    Load and return the COMPAS scores dataset.

    Returns
    -------
    dataset : dict
    """
    # Read Dataset
    df = pd.read_csv(MODULE_PATH + '/data/compas-scores-two-years.csv', delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race', 'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    def get_class(x):
        if x < 7:
            return 'Medium-Low'
        else:
            return 'High'

    df['class'] = df['decile_score'].apply(get_class)

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    columns = df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    df = df[columns]
    class_name = 'class'
    discrete = ['is_recid', 'is_violent_recid', 'two_year_recid']

    return generate_dataset(df, columns, class_name, discrete, 'compas-scores-two-years', normalize)


def load_heloc(normalize=False):
    """
    Load and return the HELOC dataset.

    Returns
    -------
    dataset : dict
    """
    # Read Dataset
    df = pd.read_csv(MODULE_PATH + '/data/heloc_dataset_v1.csv', delimiter=',')

    # Features Categorization
    columns = df.columns
    class_name = 'RiskPerformance'

    discrete = []
    return generate_dataset(df, columns, class_name, discrete, 'heloc_dataset_v1', normalize)


def load_beer(normalize=False):
    """
    Load and return the beer dataset.

    Returns
    -------
    dataset : dict
    """
    # Read Dataset
    df = pd.read_csv(MODULE_PATH + '/data/beer.csv', delimiter=',')

    # Features Categorization
    class_name = 'beer_style'
    df_cols = list(df.columns)
    df_cols.remove(class_name)
    new_cols = [class_name] + df_cols
    df = df[new_cols]

    discrete = []
    columns = df.columns
    return generate_dataset(df, columns, class_name, discrete, 'beer', normalize)


def load_pima(normalize=False):
    """
    Load and return the pima indians dataset.

    Returns
    -------
    dataset : dict
    """
    # Read Dataset
    df = pd.read_csv(MODULE_PATH + '/data/pima-indians-diabetes.csv', delimiter=',')

    # Features Categorization
    class_name = 'Class'
    df_cols = list(df.columns)
    df_cols.remove(class_name)
    new_cols = [class_name] + df_cols
    df = df[new_cols]

    discrete = []
    columns = df.columns
    return generate_dataset(df, columns, class_name, discrete, 'pima', normalize)


def load_flavia(normalize=False):
    """
    Load and return the FLAVIA dataset.

    Returns
    -------
    dataset : dict
    """
    # Read Dataset
    df = pd.read_csv(MODULE_PATH + '/data/FLAVIA3.csv', delimiter=',')

    # Features Categorization
    class_name = 'Class'
    df_cols = list(df.columns)
    df_cols.remove(class_name)
    new_cols = [class_name] + df_cols
    df = df[new_cols]

    discrete = []
    columns = df.columns
    return generate_dataset(df, columns, class_name, discrete, 'flavia', normalize)


def load_phishing(normalize=False):
    """
    Load and return the phishing dataset.

    Returns
    -------
    dataset : dict
    """
    # Read Dataset
    df = pd.read_csv(MODULE_PATH + '/data/phishing.csv', delimiter=',')
    del df['id']
    del df['PctExtResourceUrls']
    del df['PctNullSelfRedirectHyperlinks']
    del df['SubdomainLevelRT']
    del df['UrlLengthRT']
    del df['PctExtResourceUrlsRT']
    del df['AbnormalExtFormActionR']
    del df['ExtMetaScriptLinkRT']
    del df['PctExtNullSelfRedirectHyperlinksRT']

    # Features Categorization
    class_name = 'CLASS_LABEL'
    df_cols = list(df.columns)
    df_cols.remove(class_name)
    new_cols = [class_name] + df_cols
    df = df[new_cols]

    discrete = []
    columns = df.columns
    return generate_dataset(df, columns, class_name, discrete, 'phishing', normalize)


def load_iris(normalize=False):
    """
    Load and return the iris dataset.

    Returns
    -------
    dataset : dict
    """
    # Read Dataset
    iris = datasets.load_iris(as_frame=True)
    df = iris.frame

    # Features Categorization
    columns = df.columns
    class_name = columns[-1]

    df_cols = list(df.columns)
    df_cols.remove(class_name)
    new_cols = [class_name] + df_cols
    df = df[new_cols]

    discrete = []
    columns = df.columns
    return generate_dataset(df, columns, class_name, discrete, 'iris', normalize)


def load_wine(normalize=False):
    """
    Load and return the wine dataset.

    Returns
    -------
    dataset : dict
    """
    # Read Dataset
    wine = datasets.load_wine(as_frame=True)
    df = wine.frame

    # Features Categorization
    columns = df.columns
    class_name = columns[-1]

    df_cols = list(df.columns)
    df_cols.remove(class_name)
    new_cols = [class_name] + df_cols
    df = df[new_cols]

    discrete = []
    columns = df.columns
    return generate_dataset(df, columns, class_name, discrete, 'wine', normalize)


def load_breast(normalize=False):
    """
    Load and return the breast cancer dataset.

    Returns
    -------
    dataset : dict
    """
    # Read Dataset
    df = pd.read_csv(MODULE_PATH + '/data/breast.csv', delimiter=',')
    del df['id']

    # Features Categorization
    columns = df.columns
    class_name = 'diagnosis'

    discrete = []
    return generate_dataset(df, columns, class_name, discrete, 'breast', normalize)


def load_basket(normalize=False, reduced=False):
    """
    Load and return the basket dataset.

    Returns
    -------
    dataset : dict
    """
    # Read Dataset
    if reduced:
        df = pd.read_csv(MODULE_PATH + '/data/small_basket.csv', delimiter=',')
    else:
        df = pd.read_csv(MODULE_PATH + '/data/basket.csv', delimiter=',')

    # Features Categorization
    columns = df.columns
    class_name = 'Position'
    df_cols = list(df.columns)
    df_cols.remove(class_name)
    new_cols = [class_name] + df_cols
    df = df[new_cols]

    discrete = []
    return generate_dataset(df, columns, class_name, discrete, 'basket', normalize)
