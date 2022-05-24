import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix, precision_score, recall_score,classification_report, make_scorer,accuracy_score
from sklearn.model_selection import RandomizedSearchCV, learning_curve
from datetime import datetime
import dataframe_image as dfi


def model_performance(df_pp, features, technique, k, atype='', dataset='KDDCUP'):

    now = datetime.now()
    print(f'{now} - Model training started!')

    X = df_pp[features]
    y = df_pp.iloc[:,-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    scorer = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average = 'macro', zero_division = 1),
            'recall': make_scorer(recall_score, average = 'weighted'),
            'f1': make_scorer(f1_score, average = 'weighted')
                }

    param_grid = { 
            'n_estimators': [200, 300, 400],
            'max_depth' : [6,7,8,9,10],
            'criterion' : ['entropy']
    }

    clf = RandomForestClassifier(random_state = 42)

    g_search = RandomizedSearchCV(estimator = clf, 
                            param_distributions = param_grid, 
                            cv = 3, 
                            n_jobs = -1, 
                            scoring=scorer,
                            refit='f1')

    g_search.fit(X_train, y_train)

    n_estimators = g_search.best_params_['n_estimators']
    max_depth = g_search.best_params_['max_depth']
    criterion = g_search.best_params_['criterion']


    clf=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=42)
    clf.fit(X_train.values,y_train.values)
    y_pred=clf.predict(X_test)

    plot_confusion_matrix(clf, X_test, y_test)
    plt.title(f"{technique}_Confusion_Matrix")
    plt.savefig(f"visualization/Figures/{dataset}_{atype}_k{str(k)}_{str(technique)}_Confusion_Matrix.png", dpi=300, bbox_inches='tight')
    
    sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=y_pred,
    output_dict=True)

    report_df = pd.DataFrame(sk_report).transpose()
    filename = "visualization/Figures/"+dataset+'_'+atype+"_k"+str(k)+'_'+str(technique)+"_Performance_Report" + ".png"
    dfi.export(report_df,filename)

    performance_dictionary={'Technique':[], 'k':[], 'HP_n_estimators':[], 'HP_max_depth':[], 'Class':[],'Accuracy':[],'F1_Score':[],'Precision':[],'Recall':[]}

    f1 = list(f1_score(y_test, y_pred, average=None))
    matrix = confusion_matrix(y_test, y_pred)
    acc = list(matrix.diagonal()/matrix.sum(axis=1))
    pr = list(precision_score(y_test, y_pred, average=None))
    rl = list(recall_score(y_test, y_pred, average=None))

    for i in range(len(f1)):
        performance_dictionary['Technique'].append(technique)
        performance_dictionary['Class'].append(i)
        performance_dictionary['Accuracy'].append(acc[i])
        performance_dictionary['F1_Score'].append(f1[i])
        performance_dictionary['Precision'].append(pr[i])
        performance_dictionary['Recall'].append(rl[i])
        performance_dictionary['HP_n_estimators'].append(n_estimators)
        performance_dictionary['HP_max_depth'].append(max_depth)
        performance_dictionary['k'].append(k)
        

    df = pd.DataFrame.from_dict(performance_dictionary)

    train_sizes = range(1,len(X_train),7000)

    train_sizes, train_scores, validation_scores = learning_curve(
    estimator = RandomForestClassifier(),
    X = X,
    y = y, 
    train_sizes = train_sizes, 
    cv = 5,
    scoring = 'accuracy')

    now = datetime.now()
    print(f'{now} - Model performance dataframe created Successfully!')

    return df, train_scores, validation_scores



def UNSW_model_performance(df_train, df_test, features, technique, k, atype='', dataset='UNSW'):

    now = datetime.now()
    print(f'{now} - Model training started!')

    train_X =  df_train[features]
    test_X = df_test[features]
    train_y = df_train.iloc[:,-1]
    test_y = df_test.iloc[:,-1]

    scorer = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average = 'macro', zero_division = 1),
            'recall': make_scorer(recall_score, average = 'weighted'),
            'f1': make_scorer(f1_score, average = 'weighted')
                }

    param_grid = { 
            'n_estimators': [200, 300, 400],
            'max_depth' : [6,7,8,9,10],
            'criterion' : ['entropy']
    }

    clf = RandomForestClassifier(random_state = 42)

    g_search = RandomizedSearchCV(estimator = clf, 
                            param_distributions = param_grid, 
                            cv = 3, 
                            n_jobs = -1, 
                            scoring=scorer,
                            refit='f1')

    g_search.fit(train_X, train_y)

    n_estimators = g_search.best_params_['n_estimators']
    max_depth = g_search.best_params_['max_depth']
    criterion = g_search.best_params_['criterion']


    clf=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=42)
    clf.fit(train_X.values,train_y.values)
    y_pred=clf.predict(test_X)

    plot_confusion_matrix(clf, test_X, test_y)
    plt.title(f"{technique}_Confusion_Matrix")
    plt.savefig(f"visualization/Figures/{dataset}_{atype}_k{str(k)}_{str(technique)}_Confusion_Matrix.png", dpi=300, bbox_inches='tight')
    
    sk_report = classification_report(
    digits=6,
    y_true=test_y, 
    y_pred=y_pred,
    output_dict=True)

    report_df = pd.DataFrame(sk_report).transpose()
    
    filename = "visualization/Figures/"+dataset+'_'+atype+"_k"+str(k)+'_'+str(technique)+"_Performance_Report" + ".png"
    dfi.export(report_df,filename)

    performance_dictionary={'Technique':[], 'Class':[],'Accuracy':[],'F1_Score':[],'Precision':[],'Recall':[], 'HP_n_estimators':[], 'HP_max_depth':[], 'k':[]}

    f1 = list(f1_score(test_y, y_pred, average=None))
    matrix = confusion_matrix(test_y, y_pred)
    acc = list(matrix.diagonal()/matrix.sum(axis=1))
    pr = list(precision_score(test_y, y_pred, average=None))
    rl = list(recall_score(test_y, y_pred, average=None))

    for i in range(len(f1)):
        performance_dictionary['Technique'].append(technique)
        performance_dictionary['Class'].append(i)
        performance_dictionary['Accuracy'].append(acc[i])
        performance_dictionary['F1_Score'].append(f1[i])
        performance_dictionary['Precision'].append(pr[i])
        performance_dictionary['Recall'].append(rl[i])
        performance_dictionary['HP_n_estimators'].append(n_estimators)
        performance_dictionary['HP_max_depth'].append(max_depth)
        performance_dictionary['k'].append(k)
        

    df = pd.DataFrame.from_dict(performance_dictionary)

    train_sizes = range(1,int(len(train_X)*0.8),7000)

    train_sizes, train_scores, validation_scores = learning_curve(
    estimator = RandomForestClassifier(),
    X = train_X,
    y = train_y, 
    train_sizes = train_sizes, 
    cv = 5,
    scoring = 'accuracy')

    now = datetime.now()
    print(f'{now} - Model performance dataframe created Successfully!')

    return df, train_scores, validation_scores


def manual_model_performance(df_pp, features, technique, k, n_estimators, max_depth,  atype='', dataset='KDDCUP'):

    now = datetime.now()
    print(f'{now} - Model training started!')

    X = df_pp[features]
    y = df_pp.iloc[:,-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

    n_estimators = n_estimators
    max_depth = max_depth
    criterion = 'entropy'


    clf=RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion, random_state=42)
    clf.fit(X_train.values,y_train.values)
    y_pred=clf.predict(X_test)

    plot_confusion_matrix(clf, X_test, y_test)
    plt.title(f"{technique}_Confusion_Matrix")
    plt.savefig(f"visualization/Figures/{dataset}_{atype}_k{str(k)}_{str(technique)}_Confusion_Matrix.png", dpi=300, bbox_inches='tight')
    
    sk_report = classification_report(
    digits=6,
    y_true=y_test, 
    y_pred=y_pred,
    output_dict=True)

    report_df = pd.DataFrame(sk_report).transpose()
    filename = "visualization/Figures/"+dataset+'_'+atype+"_k"+str(k)+'_'+str(technique)+"_Performance_Report" + ".png"
    dfi.export(report_df,filename)

    performance_dictionary={'Technique':[], 'k':[], 'HP_n_estimators':[], 'HP_max_depth':[], 'Class':[],'Accuracy':[],'F1_Score':[],'Precision':[],'Recall':[]}

    f1 = list(f1_score(y_test, y_pred, average=None))
    matrix = confusion_matrix(y_test, y_pred)
    acc = list(matrix.diagonal()/matrix.sum(axis=1))
    pr = list(precision_score(y_test, y_pred, average=None))
    rl = list(recall_score(y_test, y_pred, average=None))

    for i in range(len(f1)):
        performance_dictionary['Technique'].append(technique)
        performance_dictionary['Class'].append(i)
        performance_dictionary['Accuracy'].append(acc[i])
        performance_dictionary['F1_Score'].append(f1[i])
        performance_dictionary['Precision'].append(pr[i])
        performance_dictionary['Recall'].append(rl[i])
        performance_dictionary['HP_n_estimators'].append(n_estimators)
        performance_dictionary['HP_max_depth'].append(max_depth)
        performance_dictionary['k'].append(k)
        

    df = pd.DataFrame.from_dict(performance_dictionary)

    return df