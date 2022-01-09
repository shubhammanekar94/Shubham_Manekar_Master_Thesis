import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, plot_confusion_matrix, precision_score, recall_score,classification_report
from datetime import datetime
import dataframe_image as dfi

def model_performance(df_pp, features, technique,k,atype='',dataset='KDDCUP'):

    X = df_pp[features]
    y = df_pp.iloc[:,-1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
    clf=RandomForestClassifier(n_estimators=200, random_state=42)
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

    performance_dictionary={'Class':[],'Accuracy':[],'F1_Score':[],'Precision':[],'Recall':[]}

    f1 = list(f1_score(y_test, y_pred, average=None))
    matrix = confusion_matrix(y_test, y_pred)
    acc = list(matrix.diagonal()/matrix.sum(axis=1))
    pr = list(precision_score(y_test, y_pred, average=None))
    rl = list(recall_score(y_test, y_pred, average=None))

    for i in range(len(f1)):

        performance_dictionary['Class'].append(i)
        performance_dictionary['Accuracy'].append(acc[i])
        performance_dictionary['F1_Score'].append(f1[i])
        performance_dictionary['Precision'].append(pr[i])
        performance_dictionary['Recall'].append(rl[i])

    df = pd.DataFrame.from_dict(performance_dictionary)

    now = datetime.now()
    print(f'{now} - Model performance dataframe created Successfully!')

    return df