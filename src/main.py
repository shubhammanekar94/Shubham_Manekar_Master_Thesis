import pandas as pd
from preprocessing import preprocessing
from features import build_features
from models import model
from visualization import visualize
import sys
from datetime import datetime
from tqdm import tqdm
import typer
import warnings

warnings.filterwarnings('ignore')

#Command line interface initialization
app = typer.Typer(help="Welcome to the CLI user manager for Intrusion Anomaly Detection Comparative Analysis!")

@app.command()
def multi_class_detection(dataset:str = typer.Argument("KDDCUP", help="Name of the Dataset", metavar="✨dataset✨")):
    """
    This module performs the Multi Class Anomaly Detection Analysis of the Attacks for the given dataset.

    """
    message_start = typer.style("Multi Class Anomaly Detection Analysis" , fg=typer.colors.GREEN, bold=True)
    message_end = typer.style(f"{dataset}", fg=typer.colors.MAGENTA, bold=True)
    message = "Strting " + message_start + " for " + message_end + " Dataset!"
    typer.echo(message)

    old_stdout = sys.stdout
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name = 'Logs/log_'+str(now).replace(' ','_').replace(':','_').replace('-','_')+'.log'
    log_file = open(file_name,"w")
    sys.stdout = log_file

    atype = 'MultiClass'
    
    print("Starting LOG!")

    raw_df = preprocessing.import_dataset(dataset)
    raw_df = preprocessing.assign_attack_types(raw_df)
    lab_enc_df = preprocessing.label_encoding(raw_df,atype,dataset)
    scaled_enc_df = preprocessing.scaling(lab_enc_df,atype,dataset)

    NOC = len(scaled_enc_df[scaled_enc_df.columns[-1]].value_counts())

    k_list = [11]

    result_final = pd.DataFrame()

    for k in tqdm(k_list):

        MI_features = build_features.MI_topk_features(scaled_enc_df, k)
        CORR_features = build_features.CORR_topk_features(scaled_enc_df, k)
        RFFI_features = build_features.RFFI_topk_features(scaled_enc_df, k)

        NumberOfInstances = 5
        SHAP_features = build_features.SHAP_topk_features(scaled_enc_df, NumberOfInstances, k)

        MI_performance, train_scores, validation_scores = model.model_performance(scaled_enc_df, MI_features, 'MI', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'MI', NOC, k,dataset)

        CORR_performance, train_scores, validation_scores = model.model_performance(scaled_enc_df, CORR_features, 'Correlation', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'Correlation', NOC, k,dataset)

        RFFI_performance, train_scores, validation_scores = model.model_performance(scaled_enc_df, RFFI_features, 'RF Feature Importance', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'RF Feature Importance', NOC, k,dataset)

        SHAP_performance, train_scores, validation_scores = model.model_performance(scaled_enc_df, SHAP_features, 'SHAP', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'SHAP', NOC, k,dataset)

        frames = [MI_performance, CORR_performance, RFFI_performance, SHAP_performance]
        result = pd.concat(frames)

        result_final = result_final.append(result, ignore_index=True)

        metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']

        for metric in metrics:
            visualize.Multi_Class_Performance_BarChart(MI_performance, CORR_performance, RFFI_performance, SHAP_performance, metric,k,dataset)
            visualize.Multi_Class_Performance_LineChart(MI_performance, CORR_performance, RFFI_performance, SHAP_performance, metric,k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Accuracy_Line.png')
        i2 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_F1_Score_Line.png')
        i3 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Precision_Line.png')
        i4 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Recall_Line.png')
        visualize.Multi_Class_Summary_Plots(i1,i2,i3,i4, 'Line_Chart',k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Accuracy_Bar.png')
        i2 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_F1_Score_Bar.png')
        i3 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Precision_Bar.png')
        i4 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Recall_Bar.png')
        visualize.Multi_Class_Summary_Plots(i1,i2,i3,i4, 'Bar_Chart',k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_MI_Confusion_Matrix.png')
        i2 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Correlation_Confusion_Matrix.png')
        i3 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_RF Feature Importance_Confusion_Matrix.png')
        i4 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_SHAP_Confusion_Matrix.png')
        visualize.Multi_Class_Summary_Plots(i1,i2,i3,i4, 'Confusion_Matrix',k,dataset)        
    
    result_final.to_csv(f'data/performance/{dataset}_{atype}.csv')
    print("Closing LOG!")
    log_file.close()

@app.command()
def binary_class_detection(dataset:str = typer.Argument("KDDCUP", help="Name of the Dataset", metavar="✨dataset✨")):
    """
    This module performs the Binary Class Anomaly Detection Analysis for the given dataset.
    """

    message_start = typer.style("Binary Class Anomaly Detection Analysis" , fg=typer.colors.GREEN, bold=True)
    message_end = typer.style(f"{dataset}", fg=typer.colors.MAGENTA, bold=True)
    message = "Strting " + message_start + " for " + message_end + " Dataset!"
    typer.echo(message)

    old_stdout = sys.stdout
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name = 'Logs/log_'+str(now).replace(' ','_').replace(':','_').replace('-','_')+'.log'
    log_file = open(file_name,"w")
    sys.stdout = log_file

    atype = 'Binary'
    NOC = 2
    print("Starting LOG!")

    raw_df = preprocessing.import_dataset(dataset)
    raw_df = preprocessing.assign_binary_target(raw_df)
    lab_enc_df = preprocessing.label_encoding(raw_df,atype,dataset)
    scaled_enc_df = preprocessing.scaling(lab_enc_df,atype,dataset)

    k_list = [32]

    result_final = pd.DataFrame()

    for k in tqdm(k_list):

        MI_features = build_features.MI_topk_features(scaled_enc_df, k)
        CORR_features = build_features.CORR_topk_features(scaled_enc_df, k)
        RFFI_features = build_features.RFFI_topk_features(scaled_enc_df, k)

        NumberOfInstances = 10
        SHAP_features = build_features.SHAP_topk_features(scaled_enc_df, NumberOfInstances, k)

        MI_performance, train_scores, validation_scores = model.model_performance(scaled_enc_df, MI_features, 'MI', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'MI', NOC, k,dataset)

        CORR_performance, train_scores, validation_scores = model.model_performance(scaled_enc_df, CORR_features, 'Correlation', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'Correlation', NOC, k,dataset)

        RFFI_performance, train_scores, validation_scores = model.model_performance(scaled_enc_df, RFFI_features, 'RF Feature Importance', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'RF Feature Importance', NOC, k,dataset)

        SHAP_performance, train_scores, validation_scores = model.model_performance(scaled_enc_df, SHAP_features, 'SHAP', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'SHAP', NOC, k,dataset)

        frames = [MI_performance, CORR_performance, RFFI_performance, SHAP_performance]
        result = pd.concat(frames)

        result_final = result_final.append(result, ignore_index=True)        

        metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']

        for metric in metrics:
            visualize.Binary_Performance_BarChart(MI_performance, CORR_performance, RFFI_performance, SHAP_performance, metric,k,dataset)
            visualize.Binary_Performance_ScatterChart(MI_performance, CORR_performance, RFFI_performance, SHAP_performance, metric,k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Accuracy_Scatter.png')
        i2 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_F1_Score_Scatter.png')
        i3 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Precision_Scatter.png')
        i4 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Recall_Scatter.png')
        visualize.Binary_Summary_Plots(i1,i2,i3,i4, 'Scatter_Chart',k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Accuracy_Bar.png')
        i2 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_F1_Score_Bar.png')
        i3 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Precision_Bar.png')
        i4 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Recall_Bar.png')
        visualize.Binary_Summary_Plots(i1,i2,i3,i4, 'Bar_Chart',k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_MI_Confusion_Matrix.png')
        i2 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Correlation_Confusion_Matrix.png')
        i3 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_RF Feature Importance_Confusion_Matrix.png')
        i4 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_SHAP_Confusion_Matrix.png')
        visualize.Binary_Summary_Plots(i1,i2,i3,i4, 'Confusion_Matrix',k,dataset)

    result_final.to_csv(f'data/performance/{dataset}_{atype}.csv')
    print("Closing LOG!")
    log_file.close()

@app.command()
def UNSW_binary_class_detection(dataset:str = typer.Argument("UNSW", help="Name of the Dataset", metavar="✨dataset✨")):
    """
    This module performs the Binary Class Anomaly Detection Analysis for the UNSW dataset.
    """

    message_start = typer.style("Binary Class Anomaly Detection Analysis for UNSW Dataset" , fg=typer.colors.GREEN, bold=True)
    message_end = typer.style(f"{dataset}", fg=typer.colors.MAGENTA, bold=True)
    message = "Strting " + message_start + " for " + message_end + " Dataset!"
    typer.echo(message)

    old_stdout = sys.stdout
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name = 'Logs/log_'+str(now).replace(' ','_').replace(':','_').replace('-','_')+'.log'
    log_file = open(file_name,"w")
    sys.stdout = log_file

    atype = 'Binary'
    NOC = 2

    print("Starting LOG!")

    df_train, df_test = preprocessing.import_UNSW(dataset='UNSW')
    df_train, df_test = preprocessing.UNSW_preprocess(df_train, df_test, dataset)


    k_list = [28,11]
    result_final = pd.DataFrame()

    for k in tqdm(k_list):

        MI_features = build_features.MI_topk_features(df_train, k)
        CORR_features = build_features.CORR_topk_features(df_train, k)
        RFFI_features = build_features.RFFI_topk_features(df_train, k)

        NumberOfInstances = 3
        SHAP_features = build_features.UNSW_SHAP_topk_features(df_train, NumberOfInstances, k)

        MI_performance, train_scores, validation_scores = model.UNSW_model_performance(df_train, df_test, MI_features, 'MI', k, atype, dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'MI', NOC, k,dataset)

        CORR_performance, train_scores, validation_scores = model.UNSW_model_performance(df_train, df_test, CORR_features, 'Correlation', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'Correlation', NOC, k,dataset)

        RFFI_performance, train_scores, validation_scores = model.UNSW_model_performance(df_train, df_test, RFFI_features, 'RF Feature Importance', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'RF Feature Importance', NOC, k,dataset)

        SHAP_performance, train_scores, validation_scores = model.UNSW_model_performance(df_train, df_test, SHAP_features, 'SHAP', k, atype,dataset)
        visualize.overfitting_graph(train_scores, validation_scores, 'SHAP', NOC, k,dataset)

        frames = [MI_performance, CORR_performance, RFFI_performance, SHAP_performance]
        result = pd.concat(frames)

        result_final = result_final.append(result, ignore_index=True) 

        metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']

        for metric in metrics:
            #visualize.Binary_Performance_BarChart(MI_performance, CORR_performance, RFFI_performance, SHAP_performance, metric,k,dataset)
            visualize.Binary_Performance_ScatterChart(MI_performance, CORR_performance, RFFI_performance, SHAP_performance, metric,k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Accuracy_Scatter.png')
        i2 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_F1_Score_Scatter.png')
        i3 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Precision_Scatter.png')
        i4 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Recall_Scatter.png')
        visualize.Binary_Summary_Plots(i1,i2,i3,i4, 'Scatter_Chart',k,dataset)

        # i1 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Accuracy_Bar.png')
        # i2 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_F1_Score_Bar.png')
        # i3 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Precision_Bar.png')
        # i4 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Recall_Bar.png')
        # visualize.Binary_Summary_Plots(i1,i2,i3,i4, 'Bar_Chart',k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_MI_Confusion_Matrix.png')
        i2 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_Correlation_Confusion_Matrix.png')
        i3 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_RF Feature Importance_Confusion_Matrix.png')
        i4 = str(f'visualization/Figures/{dataset}_Binary_k{str(k)}_SHAP_Confusion_Matrix.png')
        visualize.Binary_Summary_Plots(i1,i2,i3,i4, 'Confusion_Matrix',k,dataset)

    result_final.to_csv(f'data/performance/{dataset}_{atype}.csv')
    print("Closing LOG!")
    log_file.close()

@app.command()
def manual_multi_class_detection(dataset:str = typer.Argument("KDDCUP", help="Name of the Dataset", metavar="✨dataset✨")):
    """
    This module performs the Multi Class Anomaly Detection Analysis of the Attacks for the given dataset.

    """
    message_start = typer.style("Multi Class Anomaly Detection Analysis" , fg=typer.colors.GREEN, bold=True)
    message_end = typer.style(f"{dataset}", fg=typer.colors.MAGENTA, bold=True)
    message = "Strting " + message_start + " for " + message_end + " Dataset!"
    typer.echo(message)

    old_stdout = sys.stdout
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name = 'Logs/log_'+str(now).replace(' ','_').replace(':','_').replace('-','_')+'.log'
    log_file = open(file_name,"w")
    sys.stdout = log_file

    atype = 'MultiClass'
    
    print("Starting LOG!")

    raw_df = preprocessing.import_dataset(dataset)
    raw_df = preprocessing.assign_attack_types(raw_df)
    lab_enc_df = preprocessing.label_encoding(raw_df,atype,dataset)
    scaled_enc_df = preprocessing.scaling(lab_enc_df,atype,dataset)

    NOC = len(scaled_enc_df[scaled_enc_df.columns[-1]].value_counts())

    k_list = [28]

    result_final = pd.DataFrame()

    for k in tqdm(k_list):

        MI_features = build_features.MI_topk_features(scaled_enc_df, k)
        CORR_features = build_features.CORR_topk_features(scaled_enc_df, k)
        RFFI_features = build_features.RFFI_topk_features(scaled_enc_df, k)

        NumberOfInstances = 5
        SHAP_features = build_features.SHAP_topk_features(scaled_enc_df, NumberOfInstances, k)

        MI_performance = model.manual_model_performance(scaled_enc_df, MI_features, 'MI', k, 400, 10, atype,dataset)
        #visualize.overfitting_graph(train_scores, validation_scores, 'MI', NOC, k,dataset)

        CORR_performance = model.manual_model_performance(scaled_enc_df, CORR_features, 'Correlation', k, 400, 10, atype,dataset)
        #visualize.overfitting_graph(train_scores, validation_scores, 'Correlation', NOC, k,dataset)

        RFFI_performance = model.manual_model_performance(scaled_enc_df, RFFI_features, 'RF Feature Importance', k, 300, 9, atype,dataset)
        #visualize.overfitting_graph(train_scores, validation_scores, 'RF Feature Importance', NOC, k,dataset)

        SHAP_performance = model.manual_model_performance(scaled_enc_df, SHAP_features, 'SHAP', k, 400, 10, atype,dataset)
        #visualize.overfitting_graph(train_scores, validation_scores, 'SHAP', NOC, k,dataset)

        frames = [MI_performance, CORR_performance, RFFI_performance, SHAP_performance]
        result = pd.concat(frames)

        result_final = result_final.append(result, ignore_index=True)

        metrics = ['Accuracy', 'F1_Score', 'Precision', 'Recall']

        for metric in metrics:
            visualize.Multi_Class_Performance_BarChart(MI_performance, CORR_performance, RFFI_performance, SHAP_performance, metric,k,dataset)
            visualize.Multi_Class_Performance_LineChart(MI_performance, CORR_performance, RFFI_performance, SHAP_performance, metric,k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Accuracy_Line.png')
        i2 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_F1_Score_Line.png')
        i3 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Precision_Line.png')
        i4 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Recall_Line.png')
        visualize.Multi_Class_Summary_Plots(i1,i2,i3,i4, 'Line_Chart',k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Accuracy_Bar.png')
        i2 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_F1_Score_Bar.png')
        i3 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Precision_Bar.png')
        i4 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Recall_Bar.png')
        visualize.Multi_Class_Summary_Plots(i1,i2,i3,i4, 'Bar_Chart',k,dataset)

        i1 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_MI_Confusion_Matrix.png')
        i2 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_Correlation_Confusion_Matrix.png')
        i3 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_RF Feature Importance_Confusion_Matrix.png')
        i4 = str(f'visualization/Figures/{dataset}_MultiClass_k{str(k)}_SHAP_Confusion_Matrix.png')
        visualize.Multi_Class_Summary_Plots(i1,i2,i3,i4, 'Confusion_Matrix',k,dataset)
    
    result_final.to_csv(f'data/performance/{dataset}_{atype}.csv')
    print("Closing LOG!")
    log_file.close()


if __name__ == "__main__":
    app()