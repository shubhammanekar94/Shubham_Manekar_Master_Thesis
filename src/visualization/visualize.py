import matplotlib.pyplot as plt


def performance_viz(MI_perf_df, CORR_perf_df, RFFI_perf_df, SHAP_perf_df, metric):

    plt.rcParams['figure.dpi'] = 150
    x = ['Class 0','Class 1','Class 2','Class 3','Class 4']
    
    MI_acc = list(MI_perf_df[metric])
    CORR_acc = list(CORR_perf_df[metric])
    RFFI_acc = list(RFFI_perf_df[metric])
    SHAP_acc = list(SHAP_perf_df[metric])
    
    plt.plot(x, MI_acc, label = "Mutual Info")
    plt.plot(x, CORR_acc, label = "Correlation")
    plt.plot(x, RFFI_acc, label = "RFFI")
    plt.plot(x, SHAP_acc, label = "SHAP")

    plt.title(str.upper(metric) + ' Performance')
    plt.legend()
    plt.savefig(f"visualization/Figures/{metric}.png", dpi=300, bbox_inches='tight')