from __future__ import print_function
import numpy as np
import pandas as pd
import pickle
import sklearn
import sys
from sklearn.metrics import roc_auc_score, f1_score
from datetime import datetime, date, time

class ClientServicSubscriptioneModel:

    def __init__(self, model_location):
        with open(model_location, 'rb') as f:
            self.model = pickle.load(f)     

    def predict_score(self, X_new, df_feature):
        X_new, X_return = self.clean_and_feature_engineer(X_new, df_feature)
        return X_return, self.model.predict(X_new)

    def clean_and_feature_engineer(self, df, df_feature):

        # Drop the first column
        df.drop(columns=df.columns[0], inplace=True)
        df_feature.drop(columns=df_feature.columns[0], inplace=True)

        # Update buy_time
        df['buy_time'] = [datetime.fromtimestamp(x) for x in df['buy_time']]
        df_feature['buy_time'] = [datetime.fromtimestamp(x) for x in df_feature['buy_time']]

        # Merge train/test and feature datasets
        df = df.sort_values(by="buy_time")
        df_feature = df_feature.sort_values(by="buy_time")
        df_to_return = df.copy()
        df = pd.merge_asof(df, df_feature, by='id', on = 'buy_time', direction='nearest')

        # Remove features that have a constant value in all observations
        not_useful_features = ['75', '81', '85', '139', '203']
        df.drop(columns=not_useful_features, inplace=True)

        # Remove features that have a corr on 0.95 or higher
        multicollinear_f_to_drop = ['2', '4', '5', '14', '33', '35', '51', '71', '72', '78', '79', '112', '113', '116', '123', '124', '137', '138', '142', '151', '162', '170', '186', '217', '220']
        df.drop(columns=multicollinear_f_to_drop, inplace=True)

        # Create Descrete features
        # Features that can be divided into two groups
        f_to_split_two_groups = {'15': 3,'23': 1,'26': 0,'27': 0.5,'31': 0.5,'32': 0.5,'57': 0.5,'95': 5,'132': 0,'192': 0,'194': 0,'195': 0,'196': 0,'197': 0.5,'198': 0,'199': 0.5,'200': 0.5,'201': 0.5,'202': 0.5,'204': -0.5,'205': 0,'206': 0.5,'218': 0.5}

        # Divide features into two groups and drop the original one
        for feature, value in f_to_split_two_groups.items():
            self.split_generate_drop(feature, df, value)

        # Features that can be divided into multiple groups
        f_to_split_mult_groups = {'16': [1, 17.5, 18],'24': [1, 146, 149],'154': [0.5, 1.5, 2.5],'155': [0.5, 1.5, 2.5],'216': [0, 1]}

        # Divide features into multiple groups and drop the original one
        for feature, values in f_to_split_mult_groups.items():
            self.split_generate_drop_multiple(feature, df, values)

        # Generate new features based on the id and vas_id columns
        df['mult_attemps_mult_vas'] = (df.groupby(['id', 'vas_id'])['buy_time'].transform('count')>1).astype('int')
        df['mult_attemps'] = df['id'].duplicated().astype('int')

        # Generate new features based on buy_time
        df['buy_time_month'] = df['buy_time'].dt.month.astype('int')
        df['buy_time_week'] = df['buy_time'].dt.isocalendar().week.astype('int')
        df['buy_time_day'] = df['buy_time'].dt.isocalendar().day.astype('int')
        df.drop(columns=['buy_time'], inplace=True) 

        return df, df_to_return
    
    # Filter data in the seleted feature which are above passed value: ex: feature_data > passed value
    # Create a new feature with 0 and 1. 1 if condition == True, 0 if condition == True
    # Param: feature, data(dataframe), value_above(value to compare to)
    def generate_new_f_value_above(self, feature, data, value_above):
        new_feature_name = f'f{feature}_above_{np.absolute(value_above)}'
        data[new_feature_name] = (data[feature] > value_above).astype('int', copy=True)

    # Filter data in the seleted feature which are below passed value: ex: feature_data < passed value
    # Create a new feature with 0 and 1. 1 if condition == True, 0 if condition == True  
    # Param: feature, data(dataframe), value_below(value to compare to)
    def generate_new_f_value_below(self, feature, data, value_below):
        new_feature_name = f'f{feature}_below_{np.absolute(value_below)}'
        data[new_feature_name] = (data[feature] < value_below).astype('int', copy=True)

    # Filter data in the seleted feature which are between passed values: ex: feature_data between lower and upper
    # Create a new feature with 0 and 1. 1 if condition == True, 0 if condition == True  
    # Param: feature, data(dataframe), upper and lower(borders)    
    def generate_new_f_values_between(self, feature, data, lower, upper):
        new_feature_name = f'f{feature}_between_{np.absolute(lower)}_and_{np.absolute(upper)}'
        data[new_feature_name] = ((data[feature] < upper) & (data[feature] > lower)).astype('int')
        data[new_feature_name].astype('int')

    # Call two functions   
    def split_generate(self, feature, data, value_to_split):
        self.generate_new_f_value_above(feature, data, value_to_split)
        self.generate_new_f_value_below(feature, data, value_to_split)

    # FIlter values in the feature, generate new features and drop the original
    def split_generate_drop(self, feature, data, value_to_split):
        self.split_generate(feature, data, value_to_split)
        data.drop(columns=[feature], inplace=True) 

    # FIlter values in the feature, generate new features and drop the original   
    def split_generate_drop_multiple(self, feature, data, values_to_split):
        for i, val in enumerate(values_to_split):
            if i == 0:
                self.generate_new_f_value_below(feature, data, val)
                self.generate_new_f_values_between(feature, data,val, values_to_split[i+1])            
            elif i == (len(values_to_split)-1):
                self.generate_new_f_value_above(feature, data, val) 
            else:
                self.generate_new_f_values_between(feature, data,val, values_to_split[i+1])
        data.drop(columns=[feature], inplace=True) 
        
def main(data_location, feature_location, output_location, model_location):

    # Initialize an instance
    client_service_model = ClientServicSubscriptioneModel('final_model.pkl')

    #Load data
    df = pd.read_csv('data/data_test.csv')
    df_feature = pd.read_csv('data/features.csv', sep='\t')

    # Predict raw data
    df_return, pred = client_service_model.predict_score(df, df_feature)

    df_return['target'] = pred
    df_to_csv = df_return[['buy_time', 'id', 'vas_id', 'target']]
    df_to_csv.to_csv('data/answers_test.csv', index=None)

if __name__ == '__main__':
    main( *sys.argv[1:] )