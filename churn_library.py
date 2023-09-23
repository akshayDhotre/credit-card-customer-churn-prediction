"""
Library for churn prediction using bank's financial data.
This is used to predict the churn of customers.
"""

# import libraries
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from constants import EDA_COLUMNS, KEEP_COLUMNS, CATEGORY_COLUMNS
sns.set()

os.environ['QT_QPA_PLATFORM']='offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_df: pandas dataframe
    '''
    data_df = pd.read_csv(pth)
    return data_df


def perform_eda(data_df):
    '''
    perform eda on df and save figures to images folder
    input:
            data_df: pandas dataframe

    output:
            None
    '''
    logging.info('Starting Exploratory data analysis')
    logging.info('Dataset Dimensions: %s', data_df.shape)

    data_df['Churn'] = data_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    for column_name in EDA_COLUMNS:
        plt.figure(figsize=(20,10))
        if column_name == 'Churn':
            data_df['Churn'].hist()
        elif column_name == 'Customer_Age':
            data_df['Customer_Age'].hist()
        elif column_name == 'Marital_Status':
            data_df['Marital_Status'].value_counts('normalize').plot(kind='bar')
        elif column_name == 'Total_Trans_Ct':
            sns.histplot(data_df['Total_Trans_Ct'], stat='density', kde=True)
        elif column_name == 'heatmap':
            sns.heatmap(data_df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)

        plt.title(f'{column_name}_distribution')
        plt.savefig(f'./images/eda/{column_name}_distribution.png')
        plt.close()

    logging.info('Exploratory data analysis finished, please check figures in folder: /images/eda/')


def encoder_helper(data_df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            data_df: pandas dataframe with new columns for
    '''

    # Iterate over categorical features and add encodings for them
    for category in category_lst:
        encoded_values = []
        category_groups = data_df.groupby(category).mean()[response]

        for val in data_df[category]:
            encoded_values.append(category_groups.loc[val])
        data_df[category + '_' + response] = encoded_values

    return data_df

def perform_feature_engineering(data_df, response='Churn'):
    '''
    input:
            data_df: pandas dataframe
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data
    '''
    X = data_df[KEEP_COLUMNS]
    y = data_df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.3, random_state=42
        )

    return X_train, X_test, y_train, y_test

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    plt.text(
        0.01, 1.25,
        str('Random Forest Train'),
        {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(
        0.01, 0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(
        0.01, 0.6,
        str('Random Forest Test'),
        {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(
        0.01, 0.7,
        str(classification_report(y_train, y_train_preds_rf)),
        {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('./images/results/random_forest_classification_report.png')
    plt.close()

    plt.figure()
    plt.rc('figure', figsize=(5, 5))
    plt.text(
        0.01, 1.25,
        str('Logistic Regression Train'),
        {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(
        0.01, 0.05,
        str(classification_report(y_train, y_train_preds_lr)),
        {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(
        0.01, 0.6,
        str('Logistic Regression Test'),
        {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(
        0.01, 0.7,
        str(classification_report(y_test, y_test_preds_lr)),
        {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_regression_classification_report.png')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(f'./images/{output_pth}/random_forest_feature_importance.png')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
        }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, verbose=1)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # store ROC curve plots
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax_info = plt.gca()
    _ = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax_info, alpha=0.8)
    lrc_plot.plot(ax=ax_info, alpha=0.8)
    plt.savefig('./images/results/models_roc_plot.png')

    # save classification report images
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # save feature importance for random forest model
    feature_importance_plot(cv_rfc, X_train, 'results')

    # save best models
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

if __name__=='__main__':

    logging.info('Importing data from given Source')
    INPUT_DATAFRAME = import_data(r"./data/bank_data.csv")

    logging.info('Performing EDA on data')
    perform_eda(INPUT_DATAFRAME)

    logging.info('Encoding categorical data in input')
    DF_ENCODED = encoder_helper(INPUT_DATAFRAME,
                                category_lst=CATEGORY_COLUMNS,
                                response='Churn')

    logging.info('Performing feature engineering on data')
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        DF_ENCODED, response='Churn'
    )

    logging.info('Traing and saving the best models')
    train_models(X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)

    logging.info('Process completed!! Please check the outputs')
