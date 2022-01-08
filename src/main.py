import pandas as pd
from preprocessing import preprocessing
from features_selection import build_features
from models import model
from visualization import visualize

raw_df = preprocessing.import_kddcup()
raw_df = preprocessing.assign_attack_types(raw_df)
lab_enc_df = preprocessing.label_encoding(raw_df)
scaled_enc_df = preprocessing.scaling(lab_enc_df)

k = 20

print('Starting feature building \n')
MI_features = build_features.MI_topk_features(scaled_enc_df, k)
print('MI feature selection successful\n')

CORR_features = build_features.CORR_topk_features(scaled_enc_df, k)
print('CORR feature selection successful\n')

RFFI_features = build_features.RFFI_topk_features(scaled_enc_df, k)
print('RFFI feature selection successful\n')

NumberOfInstances = 5
SHAP_features = build_features.SHAP_topk_features(scaled_enc_df, NumberOfInstances, k)
print('SHAP feature selection successful\n')

print('Starting Model performance\n')
MI_performance = model.model_performance(scaled_enc_df, MI_features)
CORR_performance = model.model_performance(scaled_enc_df, CORR_features)
RFFI_performance = model.model_performance(scaled_enc_df, RFFI_features)
SHAP_performance = model.model_performance(scaled_enc_df, SHAP_features)

metric = 'Accuracy'

print('Starting Visualization!\n')
visualize.performance_viz(MI_performance, CORR_performance, RFFI_performance, SHAP_performance, metric)