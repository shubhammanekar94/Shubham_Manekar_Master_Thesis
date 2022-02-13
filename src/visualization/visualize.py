import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import pandas as pd
import cv2

def Multi_Class_Performance_BarChart(MI_perf_df, CORR_perf_df, RFFI_perf_df, SHAP_perf_df, metric,k,dataset='KDDCUP'):

    now = datetime.now()
    print(f'{now} - Initializing {str.upper(metric)} MultiClass comparision Bar Chart visualization..')

    plt.rcParams['figure.dpi'] = 150
    X = ['Normal','Probe','DoS','U2R','R2L']
    
    MI_acc = list(MI_perf_df[metric])
    CORR_acc = list(CORR_perf_df[metric])
    RFFI_acc = list(RFFI_perf_df[metric])
    SHAP_acc = list(SHAP_perf_df[metric])

    df = pd.DataFrame({'MI_acc': MI_acc,
                   'CORR_acc': CORR_acc,
                   'RFFI_acc': RFFI_acc,
                   'SHAP_acc': SHAP_acc
                   }, index=X)
    ax = df.plot.bar(rot=0)

    plt.title(str.upper(metric) + ' PERFORMANCE')
    plt.legend()
    plt.savefig(f"visualization/Figures/{dataset}_MultiClass_k{str(k)}_{metric}_Bar.png", dpi=300, bbox_inches='tight')

    now = datetime.now()
    print(f'{now} - {str.upper(metric)} MultiClass comparision bar chart visualization generated Successfully!')

def Multi_Class_Performance_LineChart(MI_perf_df, CORR_perf_df, RFFI_perf_df, SHAP_perf_df, metric,k,dataset):

    now = datetime.now()
    print(f'{now} - Initializing {str.upper(metric)} MultiClass comparision Line Chart visualization..')

    plt.rcParams['figure.dpi'] = 150
    X = ['Normal','Probe','DoS','U2R','R2L']
    
    SHAP_acc = list(SHAP_perf_df[metric])
    MI_acc = list(MI_perf_df[metric])
    CORR_acc = list(CORR_perf_df[metric])
    RFFI_acc = list(RFFI_perf_df[metric])
    
    df = pd.DataFrame({'CORR_acc': CORR_acc,
                   'RFFI_acc': RFFI_acc,
                   'SHAP_acc': SHAP_acc, 
                   'MI_acc': MI_acc,                
                   }, index=X)
    ax = df.plot.line()

    plt.title(str.upper(metric) + ' PERFORMANCE')
    plt.legend()
    plt.savefig(f"visualization/Figures/{dataset}_MultiClass_k{str(k)}_{metric}_Line.png", dpi=300, bbox_inches='tight')


    now = datetime.now()
    print(f'{now} - {str.upper(metric)} MultiClass comparision line chart visualization generated Successfully!')

def Multi_Class_Summary_Plots(image1,image2, image3, image4, title_text,k,dataset):
    
    now = datetime.now()
    print(f'{now} - Initializing {title_text} MultiClass summary plot..')

    plt.rcParams['figure.dpi'] = 150
    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 2
    
    Image1 = cv2.imread(image1)
    Image2 = cv2.imread(image2)
    Image3 = cv2.imread(image3)
    Image4 = cv2.imread(image4)
    
    fig.add_subplot(rows, columns, 1)
    
    plt.imshow(Image1)
    plt.axis('off')
        
    fig.add_subplot(rows, columns, 2)
    
    plt.imshow(Image2)
    plt.axis('off')
        
    fig.add_subplot(rows, columns, 3)
    
    plt.imshow(Image3)
    plt.axis('off')
        
    fig.add_subplot(rows, columns, 4)
    
    plt.imshow(Image4)
    plt.axis('off')
    
    plt.savefig(f"visualization/Figures_Summary/{dataset}_MultiClass_k{str(k)}_{title_text}_Summary Plot.png", dpi=300, bbox_inches='tight')

    now = datetime.now()
    print(f'{now} - {title_text} MultiClass summary plot completed!')

def Binary_Performance_BarChart(MI_perf_df, CORR_perf_df, RFFI_perf_df, SHAP_perf_df, metric,k,dataset='KDDCUP'):

    now = datetime.now()
    print(f'{now} - Initializing {str.upper(metric)} Binary comparision Bar Chart visualization..')

    plt.rcParams['figure.dpi'] = 150
    X = ['Normal','Anomaly']
    
    MI_acc = list(MI_perf_df[metric])
    CORR_acc = list(CORR_perf_df[metric])
    RFFI_acc = list(RFFI_perf_df[metric])
    SHAP_acc = list(SHAP_perf_df[metric])

    df = pd.DataFrame({'MI_acc': MI_acc,
                   'CORR_acc': CORR_acc,
                   'RFFI_acc': RFFI_acc,
                   'SHAP_acc': SHAP_acc
                   }, index=X)
    ax = df.plot.bar(rot=0)

    plt.title(str.upper(metric) + ' PERFORMANCE')
    plt.legend()
    plt.savefig(f"visualization/Figures/{dataset}_Binary_k{str(k)}_{metric}_Bar.png", dpi=300, bbox_inches='tight')

    now = datetime.now()
    print(f'{now} - {str.upper(metric)} BinaryClass comparision bar chart visualization generated Successfully!')

def Binary_Performance_ScatterChart(MI_perf_df, CORR_perf_df, RFFI_perf_df, SHAP_perf_df, metric,k,dataset):

    now = datetime.now()
    print(f'{now} - Initializing {str.upper(metric)} BinaryClass comparision Scatter Chart visualization..')

    plt.rcParams['figure.dpi'] = 150
    
    X = ['Normal','Anomaly']
    
    SHAP = list(SHAP_perf_df[metric])
    MI = list(MI_perf_df[metric])
    CORR = list(CORR_perf_df[metric])
    RFFI = list(RFFI_perf_df[metric])
    
    df = pd.DataFrame({'CORR': CORR,
                   'RFFI': RFFI,
                   'SHAP': SHAP, 
                   'MI': MI,
                   'Class' : X
                   })
    c=['r','g','orange', 'purple']
    Y= ['CORR','RFFI','SHAP','MI']
    fig, ax = plt.subplots()

    for m in range(len(Y)):
        ax.scatter(x=df['Class'], y=df[Y[m]], color=c[m],label=df[Y[m]].name)    

    plt.xlabel('Class')
    plt.ylabel(metric)
    plt.title(str.upper(metric) + ' PERFORMANCE')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"visualization/Figures/{dataset}_Binary_k{str(k)}_{metric}_Scatter.png", dpi=300, bbox_inches='tight')


    now = datetime.now()
    print(f'{now} - {str.upper(metric)} BinaryClass comparision line chart visualization generated Successfully!')

def Binary_Summary_Plots(image1,image2, image3, image4, title_text,k,dataset):
    
    now = datetime.now()
    print(f'{now} - Initializing {title_text} BinaryClass summary plot..')

    plt.rcParams['figure.dpi'] = 150
    fig = plt.figure(figsize=(10, 7))
    rows = 2
    columns = 2
    
    Image1 = cv2.imread(image1)
    Image2 = cv2.imread(image2)
    Image3 = cv2.imread(image3)
    Image4 = cv2.imread(image4)
    
    fig.add_subplot(rows, columns, 1)
    
    plt.imshow(Image1)
    plt.axis('off')
        
    fig.add_subplot(rows, columns, 2)
    
    plt.imshow(Image2)
    plt.axis('off')
        
    fig.add_subplot(rows, columns, 3)
    
    plt.imshow(Image3)
    plt.axis('off')
        
    fig.add_subplot(rows, columns, 4)
    
    plt.imshow(Image4)
    plt.axis('off')
    
    plt.savefig(f"visualization/Figures_Summary/{dataset}_Binary_k{str(k)}_{title_text}_Summary_Plot.png", dpi=300, bbox_inches='tight')

    now = datetime.now()
    print(f'{now} - {title_text} BinaryClass summary plot completed!')

def UNSW_Binary_Performance(MI_perf_df, CORR_perf_df, RFFI_perf_df, SHAP_perf_df, metric,k,dataset):
    
    now = datetime.now()
    print(f'{now} - Initializing {str.upper(metric)} BinaryClass comparision Scatter Chart visualization..')

    plt.rcParams['figure.dpi'] = 150
    
    X = ['Normal','Anomaly']
    
    SHAP = list(SHAP_perf_df[metric])
    MI = list(MI_perf_df[metric])
    CORR = list(CORR_perf_df[metric])
    RFFI = list(RFFI_perf_df[metric])
    
    df = pd.DataFrame({'CORR': CORR,
                   'RFFI': RFFI,
                   'SHAP': SHAP, 
                   'MI': MI,
                   'Class' : X
                   })
    c=['r','g','orange', 'purple']
    Y= ['CORR','RFFI','SHAP','MI']
    fig, ax = plt.subplots()

    for m in range(len(Y)):
        ax.scatter(x=df['Class'], y=df[Y[m]], color=c[m],label=df[Y[m]].name)    

    plt.xlabel('Class')
    plt.ylabel(metric)
    plt.title(str.upper(metric) + ' PERFORMANCE')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"visualization/Figures/{dataset}_Binary_k{str(k)}_{metric}_Scatter.png", dpi=300, bbox_inches='tight')


    now = datetime.now()
    print(f'{now} - {str.upper(metric)} BinaryClass comparision line chart visualization generated Successfully!')

def overfitting_graph(train_scores, validation_scores, technique, NOC, k,dataset):
    
    now = datetime.now()
    print(f'{now} - Initializing {str.upper(technique)} {NOC} - Class Overfitting visualization..')

    plt.rcParams['figure.dpi'] = 150

    length = list(range(len(train_scores)))
    x_axis = [x*7000 for x in length]

    for i in range(NOC):

        y_axis_train = train_scores[:,i]
        y_axis_val = validation_scores[:,i]

        df = pd.DataFrame({'Training Accuracy': y_axis_train,
                    'Validation Accuracy': y_axis_val
                    }, index=x_axis)
        ax = df.plot.line()

        plt.title(f' {str.upper(technique)} CLASS {str(i)} PERFORMANCE')
        plt.xlabel('Training Sizes')
        plt.ylabel('Accuracy Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"visualization/Figures/{dataset}_{str.upper(technique)}_class_{i}_k{str(k)}_overfitting.png", dpi=300, bbox_inches='tight')


    now = datetime.now()
    print(f'{now} - {str.upper(technique)} {NOC} - Class Overfitting visualization generated Successfully!')