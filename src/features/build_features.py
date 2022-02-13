import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap
from datetime import datetime

def MI_topk_features(df_pp, k):

    now = datetime.now()
    print(f'{now} - Mutual Information feature selection Initialized..')

    X = df_pp.iloc[:,:-1]
    y = df_pp.iloc[:,-1]

    importances = mutual_info_classif(X,y)
    feature_imp = pd.Series(importances, df_pp.columns[0:len(df_pp.columns)-1])

    MI_features = list(feature_imp.sort_values(ascending = False)[:k].index)

    now = datetime.now()
    print(f'{now} - Mutual Information feature selection Successful!')

    return MI_features

def CORR_topk_features(df_pp, k):

    now = datetime.now()
    print(f'{now} - Correlation feature selection Initialized..')

    target = list(df_pp.columns)[-1]

    cor = df_pp.corr()
    CORR_features = list(abs(cor[target]).sort_values(ascending=False)[:k+1].index)
    CORR_features.remove(target)

    now = datetime.now()
    print(f'{now} - Correlation feature selection Successful!')

    return CORR_features

def RFFI_topk_features(df_pp, k):
    
    now = datetime.now()
    print(f'{now} - RF Feature Importance feature selection Initialized..')

    X = df_pp.iloc[:,:-1]
    y = df_pp.iloc[:,-1]

    rf = RandomForestClassifier(n_estimators=200,random_state=42)
    rf.fit(X.values,y.values)
    rffi = rf.feature_importances_

    features = pd.Series(rffi, df_pp.columns[0:len(df_pp.columns)-1])
    RFFI_features = list(features.sort_values(ascending = False)[:k].index)

    now = datetime.now()
    print(f'{now} - RF Feature Importance feature selection Successful!')

    return RFFI_features

def SHAP_topk_features(df_pp, number_of_instances, k):

    now = datetime.now()
    print(f'{now} - SHAP feature selection Initialized!')

    X = df_pp.iloc[:,:-1]
    y = df_pp.iloc[:,-1]

    train_X, test_X, train_y, test_y = train_test_split(X,y, random_state=42)
    rf = RandomForestClassifier(random_state=42).fit(train_X.values, train_y.values)
    shap_df = pd.DataFrame()

    for target in list(test_y.value_counts().index):   
        i = []
        v = []

        for index, value in zip(range(len(test_y)), test_y):    
            i.append(index)
            v.append(value)
        
        y_val = pd.Series(v,i)
        y_val = list(y_val[y_val==target].index)

        idx_threshold = number_of_instances
        final_idx = []

        for idx in y_val:
            if idx_threshold > 0:
                row_instance = test_X.iloc[idx]
                row_pred = row_instance.values.reshape(1, -1)

                if float(rf.predict_proba(row_pred)[0][target]) == 1.0:
                    final_idx.append(idx)
                    idx_threshold = idx_threshold - 1
            else:
                break

        temp_df = pd.DataFrame()

        for idx in final_idx:
            row_instance = test_X.iloc[idx]
            row_pred = row_instance.values.reshape(1, -1)
            
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(row_instance)

            series_idx = row_instance.index
            series_val = shap_values[target]

            feature_series = pd.Series(series_val, series_idx)
            
            shap_df_temp = feature_series.to_frame().rename(columns={0:'shap_value'}).rename_axis('features').reset_index()
            temp_df = temp_df.append(shap_df_temp, ignore_index = True)

        temp_df = temp_df.sort_values(by=['shap_value'], ascending=False).drop_duplicates(subset='features', keep='first')
        shap_df = shap_df.append(temp_df, ignore_index = True)    

    shap_df = shap_df.sort_values(by=['shap_value'], ascending=False).drop_duplicates(subset='features', keep='first')
    shap_features = list(shap_df['features'].iloc[:k])

    now = datetime.now()
    print(f'{now} - SHAP feature selection Successful!')

    return shap_features


def UNSW_SHAP_topk_features(df_train, number_of_instances, k):

    now = datetime.now()
    print(f'{now} - SHAP feature selection Initialized!')

    train_X =  df_train.iloc[:,:-1]
    train_y = df_train.iloc[:,-1]

    rf = RandomForestClassifier(random_state=42).fit(train_X.values,train_y.values)
    shap_df = pd.DataFrame()

    for target in list(train_y.value_counts().index):   
        target = int(target)
        i = []
        v = []

        for index, value in zip(range(len(train_y)), train_y):    
            i.append(index)
            v.append(value)
        
        y_val = pd.Series(v,i)
        y_val = list(y_val[y_val==target].index)

        idx_threshold = number_of_instances
        final_idx = []

        for idx in y_val:
            if idx_threshold > 0:
                row_instance = train_X.iloc[idx]
                row_pred = row_instance.values.reshape(1, -1)

                if float(rf.predict_proba(row_pred)[0][target]) == 1.0:
                    final_idx.append(idx)
                    idx_threshold = idx_threshold - 1
            else:
                break

        temp_df = pd.DataFrame()

        for idx in final_idx:
            row_instance = train_X.iloc[idx]
            row_pred = row_instance.values.reshape(1, -1)
            
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(row_instance)

            series_idx = row_instance.index
            series_val = shap_values[target]

            feature_series = pd.Series(series_val, series_idx)
            
            shap_df_temp = feature_series.to_frame().rename(columns={0:'shap_value'}).rename_axis('features').reset_index()
            temp_df = temp_df.append(shap_df_temp, ignore_index = True)

        temp_df = temp_df.sort_values(by=['shap_value'], ascending=False).drop_duplicates(subset='features', keep='first')
        shap_df = shap_df.append(temp_df, ignore_index = True)    

    shap_df = shap_df.sort_values(by=['shap_value'], ascending=False).drop_duplicates(subset='features', keep='first')
    shap_features = list(shap_df['features'].iloc[:k])

    now = datetime.now()
    print(f'{now} - SHAP feature selection Successful!')

    return shap_features