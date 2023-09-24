"""
Script for testing churn_library.py using pytest

Author: Akshay Dhotre
Date: September 22, 2023
"""
import os
import logging
import joblib
import pandas as pd
import pytest
import churn_library as cls
from constants import DATA_PATH, \
    CATEGORY_COLUMNS, EDA_IMAGES, RESULT_IMAGES, EDA_IMAGES_PATH, RESULT_IMAGES_PATH

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope='module', name='data_path')
def path():
    '''
    Fixture for returning file path to test import_data
    '''
    return DATA_PATH


@pytest.fixture(scope='module', name='input_dataframe')
def input_df():
    '''
    pytest fixture to create dataframe from csv file
    '''
    try:
        data_df = cls.import_data(DATA_PATH)
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    return data_df


@pytest.fixture(scope='module', name='encoded_dataframe')
def encoded_df(input_dataframe):
    '''
    pytest fixture to create encoded dataframe from input dataframe
    '''
    encoded_data_df = cls.encoder_helper(input_dataframe,
                                         category_lst=CATEGORY_COLUMNS,
                                         response='Churn')
    logging.info('Successfully generated encoded dataframe')
    return encoded_data_df


@pytest.fixture(scope='module', name='processed_dataframe')
def processed_df(encoded_dataframe):
    '''
    pytest fixture to get dataframe after feature engineering
    '''
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        encoded_dataframe, response='Churn'
    )
    logging.info('Successfully completed feature engioneering on dataframe!')

    return x_train, x_test, y_train, y_test


def test_import_data(data_path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_df = cls.import_data(data_path)
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert data_df.shape[0] > 0
        assert data_df.shape[1] > 0
        logging.info('Testing import_data: input dataframe is as expected!')
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't have rows or columns")
        raise err


def test_eda(input_dataframe):
    '''
    test perform eda function
    '''
    cls.perform_eda(input_dataframe)

    for image_name in EDA_IMAGES:
        image_file_path = EDA_IMAGES_PATH + image_name
        try:
            assert os.path.isfile(image_file_path)

        except AssertionError as err:
            logging.error(
                'Testing perform_eda, image %s not found',
                image_name)
            raise err

    logging.info('Testing perform_eda: Successfully EDA completed!')


def test_encoder_helper(encoded_dataframe):
    '''
    test encoder helper
    '''
    try:
        assert isinstance(encoded_dataframe, pd.DataFrame)
        for column_name in CATEGORY_COLUMNS:
            assert column_name + '_Churn' in encoded_dataframe.columns
    except AssertionError as err:
        logging.error('Testing encoder_helper: error with dataframe')
        raise err

    logging.info('Testing encoder_helper: found required column!')


def test_perform_feature_engineering(processed_dataframe):
    '''
    test perform_feature_engineering
    '''
    x_train = processed_dataframe[0]
    x_test = processed_dataframe[1]
    y_train = processed_dataframe[2]
    y_test = processed_dataframe[3]

    try:
        assert len(x_test) == len(y_test)
        assert len(x_train) == len(y_train)
        assert len(x_train) > len(x_test)
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: problem with test and train data')
        raise err


def test_train_models(processed_dataframe):
    '''
    test train_models
    '''
    x_train = processed_dataframe[0]
    x_test = processed_dataframe[1]
    y_train = processed_dataframe[2]
    y_test = processed_dataframe[3]

    cls.train_models(x_train, x_test, y_train, y_test)

    try:
        joblib.load('./models/rfc_model.pkl')
        joblib.load('./models/logistic_model.pkl')
        logging.info('Testing train_models: found required models')
    except FileNotFoundError as err:
        logging.error('Testing train_models: cannot find models')
        raise err

    for image_name in RESULT_IMAGES:
        image_file_path = RESULT_IMAGES_PATH + image_name
        try:
            assert os.path.isfile(image_file_path)
        except AssertionError as err:
            logging.error(
                'Testing train_models, image %s not found', image_name)
            raise err

    logging.info('Testing train_models: completed model creation!')


if __name__ == "__main__":
    for directory in [
        "./logs",
        "./images/eda",
        "./images/results",
            "./models"]:
        for root, dirs, files in os.walk(directory):
            for file in files:
                full_file_path = os.path.join(root, file)
                os.remove(full_file_path)
