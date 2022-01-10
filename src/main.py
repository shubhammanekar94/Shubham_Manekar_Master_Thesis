import pandas as pd
from preprocessing import preprocessing
from features import build_features
from models import model
from visualization import visualize
import sys
from datetime import datetime
from tqdm import tqdm


def multi_class_detection(dataset):

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

    k_list = [28,21,12]

    for k in tqdm(k_list):

        MI_features = build_features.MI_topk_features(scaled_enc_df, k)
        CORR_features = build_features.CORR_topk_features(scaled_enc_df, k)
        RFFI_features = build_features.RFFI_topk_features(scaled_enc_df, k)

        NumberOfInstances = 10
        SHAP_features = build_features.SHAP_topk_features(scaled_enc_df, NumberOfInstances, k)

        MI_performance = model.model_performance(scaled_enc_df, MI_features, 'MI', k, atype,dataset)
        CORR_performance = model.model_performance(scaled_enc_df, CORR_features, 'Correlation', k, atype,dataset)
        RFFI_performance = model.model_performance(scaled_enc_df, RFFI_features, 'RF Feature Importance', k, atype,dataset)
        SHAP_performance = model.model_performance(scaled_enc_df, SHAP_features, 'SHAP', k, atype,dataset)

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

    print("Closing LOG!")
    log_file.close()

def binary_class_detection(dataset):

    old_stdout = sys.stdout
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_name = 'Logs/log_'+str(now).replace(' ','_').replace(':','_').replace('-','_')+'.log'
    log_file = open(file_name,"w")
    sys.stdout = log_file

    atype = 'Binary'

    print("Starting LOG!")

    raw_df = preprocessing.import_dataset(dataset)
    raw_df = preprocessing.assign_binary_target(raw_df)
    lab_enc_df = preprocessing.label_encoding(raw_df,atype,dataset)
    scaled_enc_df = preprocessing.scaling(lab_enc_df,atype,dataset)

    k_list = [28,21,12]

    for k in tqdm(k_list):

        MI_features = build_features.MI_topk_features(scaled_enc_df, k)
        CORR_features = build_features.CORR_topk_features(scaled_enc_df, k)
        RFFI_features = build_features.RFFI_topk_features(scaled_enc_df, k)

        NumberOfInstances = 10
        SHAP_features = build_features.SHAP_topk_features(scaled_enc_df, NumberOfInstances, k)

        MI_performance = model.model_performance(scaled_enc_df, MI_features, 'MI', k, atype,dataset)
        CORR_performance = model.model_performance(scaled_enc_df, CORR_features, 'Correlation', k, atype,dataset)
        RFFI_performance = model.model_performance(scaled_enc_df, RFFI_features, 'RF Feature Importance', k, atype,dataset)
        SHAP_performance = model.model_performance(scaled_enc_df, SHAP_features, 'SHAP', k, atype,dataset)

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

    print("Closing LOG!")
    log_file.close()



dataset = 'KDDCUP'
binary_class_detection(dataset)
multi_class_detection(dataset)

dataset = 'NSLKDD'
binary_class_detection(dataset)