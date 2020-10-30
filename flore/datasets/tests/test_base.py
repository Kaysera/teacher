import pytest

from flore.datasets import load_german, load_adult, load_compas, load_heloc
from os.path import dirname, join


def check_data(load_data, dataset_values):
    """ Check the different values of the dataset to load """

    dataset = load_data()

    assert dataset['df'].shape == dataset_values['dfshape']
    assert dataset['name'] == dataset_values['name']
    assert dataset['columns'] == dataset_values['columns']

    assert dataset['class_name'] == dataset_values['class_name']
    assert dataset['possible_outcomes'] == dataset_values['possible_outcomes']
    assert dataset['type_features'] == dataset_values['type_features']

    assert dataset['features_type'] == dataset_values['features_type']

    assert dataset['continuous'] == dataset_values['continuous']
    assert dataset['idx_features'] == dataset_values['idx_features']

    assert set(dataset['discrete']) == dataset_values['discrete']

    assert dataset['X'].shape == dataset_values['Xshape']
    assert dataset['y'].shape == dataset_values['yshape']

def test_load_german():

    dataset_values = {
        'dfshape' : (1000, 21),
        'name' : 'german_credit',
        'columns' : ['default', 'account_check_status', 'duration_in_month', 'credit_history', 'purpose', 'credit_amount', 'savings', 'present_emp_since', 'installment_as_income_perc', 'personal_status_sex', 'other_debtors', 'present_res_since', 'property', 'age', 'other_installment_plans', 'housing', 'credits_this_bank', 'job', 'people_under_maintenance', 'telephone', 'foreign_worker'],
        'class_name' : 'default',
        'possible_outcomes' : [0,1],
        'type_features' : {'integer': ['default', 'duration_in_month', 'credit_amount', 'installment_as_income_perc', 'present_res_since', 'age', 'credits_this_bank', 'people_under_maintenance'], 'double': [], 'string': ['account_check_status', 'credit_history', 'purpose', 'savings', 'present_emp_since', 'personal_status_sex', 'other_debtors', 'property', 'other_installment_plans', 'housing', 'job', 'telephone', 'foreign_worker']},
        'features_type' : {'default': 'integer', 'duration_in_month': 'integer', 'credit_amount': 'integer', 'installment_as_income_perc': 'integer', 'present_res_since': 'integer', 'age': 'integer', 'credits_this_bank': 'integer', 'people_under_maintenance': 'integer', 'account_check_status': 'string', 'credit_history': 'string', 'purpose': 'string', 'savings': 'string', 'present_emp_since': 'string', 'personal_status_sex': 'string', 'other_debtors': 'string', 'property': 'string', 'other_installment_plans': 'string', 'housing': 'string', 'job': 'string', 'telephone': 'string', 'foreign_worker': 'string'},
        'continuous' : ['duration_in_month', 'credit_amount', 'age'],
        'idx_features' : {0: 'account_check_status', 1: 'duration_in_month', 2: 'credit_history', 3: 'purpose', 4: 'credit_amount', 5: 'savings', 6: 'present_emp_since', 7: 'installment_as_income_perc', 8: 'personal_status_sex', 9: 'other_debtors', 10: 'present_res_since', 11: 'property', 12: 'age', 13: 'other_installment_plans', 14: 'housing', 15: 'credits_this_bank', 16: 'job', 17: 'people_under_maintenance', 18: 'telephone', 19: 'foreign_worker'},
        'discrete' : set(['savings', 'job', 'credit_history', 'housing', 'foreign_worker', 'property', 'credits_this_bank', 'purpose', 'installment_as_income_perc', 'people_under_maintenance', 'personal_status_sex', 'other_debtors', 'telephone', 'other_installment_plans', 'account_check_status', 'present_res_since', 'present_emp_since', 'default']),
        'Xshape' : (1000, 20),
        'yshape' : (1000, )
    }

    check_data(load_german, dataset_values)

def test_load_adult():

    dataset_values = {
        'name': 'adult', 
        'columns': ['class', 'age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'], 
        'class_name': 'class', 
        'possible_outcomes': ['<=50K', '>50K'], 
        'type_features': {'integer': ['age', 'capital-gain', 'capital-loss', 'hours-per-week'], 'double': [], 'string': ['class', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']}, 
        'features_type': {'age': 'integer', 'capital-gain': 'integer', 'capital-loss': 'integer', 'hours-per-week': 'integer', 'class': 'string', 'workclass': 'string', 'education': 'string', 'marital-status': 'string', 'occupation': 'string', 'relationship': 'string', 'race': 'string', 'sex': 'string', 'native-country': 'string'}, 
        'discrete': set(['native-country', 'relationship', 'workclass', 'marital-status', 'education', 'sex', 'class', 'occupation', 'race']), 
        'continuous': ['age', 'capital-gain', 'capital-loss', 'hours-per-week'], 
        'idx_features': {0: 'age', 1: 'workclass', 2: 'education', 3: 'marital-status', 4: 'occupation', 5: 'relationship', 6: 'race', 7: 'sex', 8: 'capital-gain', 9: 'capital-loss', 10: 'hours-per-week', 11: 'native-country'}, 
        'dfshape': (32561, 13), 
        'Xshape': (32561, 12), 
        'yshape': (32561,)
    }

    check_data(load_adult, dataset_values)


def test_load_compas():
    
    dataset_values = {
        'name': 'compas-scores-two-years', 
        'columns': ['class', 'age', 'age_cat', 'sex', 'race', 'priors_count', 'days_b_screening_arrest', 'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'length_of_stay'], 
        'class_name': 'class', 
        'possible_outcomes': ['Medium-Low', 'High'], 
        'type_features': {'integer': ['age', 'priors_count', 'is_recid', 'is_violent_recid', 'two_year_recid'], 'double': [], 'string': ['class', 'age_cat', 'sex', 'race', 'c_charge_degree']}, 
        'features_type': {'age': 'integer', 'priors_count': 'integer', 'is_recid': 'integer', 'is_violent_recid': 'integer', 'two_year_recid': 'integer', 'class': 'string', 'age_cat': 'string', 'sex': 'string', 'race': 'string', 'c_charge_degree': 'string'}, 
        'discrete': set(['age_cat', 'two_year_recid', 'class', 'sex', 'is_recid', 'is_violent_recid', 'c_charge_degree', 'race']), 
        'continuous': ['age', 'priors_count'], 
        'idx_features': {0: 'age', 1: 'age_cat', 2: 'sex', 3: 'race', 4: 'priors_count', 5: 'days_b_screening_arrest', 6: 'c_charge_degree', 7: 'is_recid', 8: 'is_violent_recid', 9: 'two_year_recid', 10: 'length_of_stay'}, 
        'dfshape': (7214, 12), 
        'Xshape': (7214, 11), 
        'yshape': (7214,)
    }
    check_data(load_compas, dataset_values)


def test_load_heloc():
    
    dataset_values  = {
        'name': 'heloc_dataset_v1', 
        'columns': ['RiskPerformance', 'ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance'], 
        'class_name': 'RiskPerformance', 
        'possible_outcomes': ['Bad', 'Good'], 
        'type_features': {'integer': ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance'], 'double': [], 'string': ['RiskPerformance']}, 
        'features_type': {'ExternalRiskEstimate': 'integer', 'MSinceOldestTradeOpen': 'integer', 'MSinceMostRecentTradeOpen': 'integer', 'AverageMInFile': 'integer', 'NumSatisfactoryTrades': 'integer', 'NumTrades60Ever2DerogPubRec': 'integer', 'NumTrades90Ever2DerogPubRec': 'integer', 'PercentTradesNeverDelq': 'integer', 'MSinceMostRecentDelq': 'integer', 'MaxDelq2PublicRecLast12M': 'integer', 'MaxDelqEver': 'integer', 'NumTotalTrades': 'integer', 'NumTradesOpeninLast12M': 'integer', 'PercentInstallTrades': 'integer', 'MSinceMostRecentInqexcl7days': 'integer', 'NumInqLast6M': 'integer', 'NumInqLast6Mexcl7days': 'integer', 'NetFractionRevolvingBurden': 'integer', 'NetFractionInstallBurden': 'integer', 'NumRevolvingTradesWBalance': 'integer', 'NumInstallTradesWBalance': 'integer', 'NumBank2NatlTradesWHighUtilization': 'integer', 'PercentTradesWBalance': 'integer', 'RiskPerformance': 'string'}, 
        'discrete': set(['RiskPerformance']), 
        'continuous': ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen', 'AverageMInFile', 'NumSatisfactoryTrades', 'NumTrades60Ever2DerogPubRec', 'NumTrades90Ever2DerogPubRec', 'PercentTradesNeverDelq', 'MSinceMostRecentDelq', 'MaxDelq2PublicRecLast12M', 'MaxDelqEver', 'NumTotalTrades', 'NumTradesOpeninLast12M', 'PercentInstallTrades', 'MSinceMostRecentInqexcl7days', 'NumInqLast6M', 'NumInqLast6Mexcl7days', 'NetFractionRevolvingBurden', 'NetFractionInstallBurden', 'NumRevolvingTradesWBalance', 'NumInstallTradesWBalance', 'NumBank2NatlTradesWHighUtilization', 'PercentTradesWBalance'], 
        'idx_features': {0: 'ExternalRiskEstimate', 1: 'MSinceOldestTradeOpen', 2: 'MSinceMostRecentTradeOpen', 3: 'AverageMInFile', 4: 'NumSatisfactoryTrades', 5: 'NumTrades60Ever2DerogPubRec', 6: 'NumTrades90Ever2DerogPubRec', 7: 'PercentTradesNeverDelq', 8: 'MSinceMostRecentDelq', 9: 'MaxDelq2PublicRecLast12M', 10: 'MaxDelqEver', 11: 'NumTotalTrades', 12: 'NumTradesOpeninLast12M', 13: 'PercentInstallTrades', 14: 'MSinceMostRecentInqexcl7days', 15: 'NumInqLast6M', 16: 'NumInqLast6Mexcl7days', 17: 'NetFractionRevolvingBurden', 18: 'NetFractionInstallBurden', 19: 'NumRevolvingTradesWBalance', 20: 'NumInstallTradesWBalance', 21: 'NumBank2NatlTradesWHighUtilization', 22: 'PercentTradesWBalance'}, 
        'dfshape': (10459, 24), 
        'Xshape': (10459, 23), 
        'yshape': (10459,)
    }    

    check_data(load_heloc, dataset_values)