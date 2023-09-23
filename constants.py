"""
Module containing all constatnts used in churn_library
"""
DATA_PATH = r'./data/bank_data.csv'
EDA_IMAGES_PATH = r'./images/eda/'
RESULT_IMAGES_PATH = r'./images/results/'

CATEGORY_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANTITY_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

EDA_COLUMNS = [
    "Churn",
    "Customer_Age",
    "Marital_Status",
    'Total_Trans_Ct',
    "heatmap"
]

KEEP_COLUMNS = [
    'Customer_Age', 'Dependent_count', 'Months_on_book',
    'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
    'Income_Category_Churn', 'Card_Category_Churn'
]

EDA_IMAGES = [
    'Churn_distribution.png',
    'Customer_Age_distribution.png',
    'Marital_Status_distribution.png',
    'Total_Trans_Ct_distribution.png',
    'heatmap_distribution.png'
]

RESULT_IMAGES = [
    'models_roc_plot.png',
    'random_forest_feature_importance.png',
    'random_forest_classification_report.png',
    'logistic_regression_classification_report.png',
]
