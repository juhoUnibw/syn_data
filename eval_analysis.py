from unittest.mock import inplace

import pandas as pd
import argparse
import scipy.stats as stats
import matplotlib.pyplot as plt
from importlib import import_module, reload
preprocessing = import_module('data.preprocessing')
import numpy as np
from pkg_resources import require
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from torch.utils.data.datapipes.dataframe.dataframe_wrapper import concat

# available datasets and paths
dataset_names = \
    {
        'breast_cancer': 'Breast Cancer/breast-cancer.data.txt',
        'breast_tissue': 'Breast Tissue/breast_tissue.csv',
        'cardiotocography': 'Cardiotocography/CTG_clean.csv',
        'kidney': 'Chronic Kidney Disease/chronic_kidney_disease.arff.txt',
        'dermatology': 'Dermatology/dermatology.data.txt',
        'diabetes': 'Diabetes/diabetes.csv',
        'retinography': 'Diabetic Retinography/messidor_features.arff.txt',
        'echocardiogram': 'Echocardiogram/echocardiogram.csv',
        'heart': 'Heart Disease/heart_cleveland.csv',
        'lymphography': 'Lymphography/lymphography.csv',
        'patient': 'Postoperative Patient Data/post-operative.data.txt',
        'stroke': 'Stroke/healthcare-dataset-stroke-data.csv',
        'thoracic_surgery': 'Thoracic Surgery/ThoraricSurgery.arff.txt',
        'thyroid': 'Thyroid Disease/thyroid.csv',
        'tumor': 'Tumor/primary-tumor.data.txt',
        'sani': 'Z-Adlidazeh Sani/sani.xlsx',
        'eye': 'EEG Eye State/EEG Eye State.arff.txt'}

'''
Dataset Name [source] # Records # Features # Num. # Cat. # Classes Missing Values
'''
data_sizes={
'Diabetes': 768,
'Heart': 297,
'Breast Cancer': 286,
'Dermatology': 366,
'Sani': 303,
'Breast Tissue': 106,
'Kidney': 768,
'Cardiotocography': 2126,
'Retinography': 1152,
'Echocardiogram': 132,
'Eye': 14980,
'Lymphography': 148,
'Patient': 90,
'Tumor': 339,
'Stroke': 5110,
'Thoracic Surgery': 470,
'Thyroid': 9172,
}


# available methods
methods = ['tvae', 'gausscop', 'ctgan', 'arf', 'nflow', 'knnmtd', 'mcgen', 'corgan',  'smote',
           'priv_bayes', 'cart'] #'great', 'tabula', 'ensgen', 'genary',


def load_data(dataset_name):
    reload(preprocessing)
    dataset, class_var, cat_feat_names, num_feat_names = preprocessing.load_data(dataset_name)

    return dataset, class_var, cat_feat_names, num_feat_names

# data synthesis process
def calc_std(data, method):

    std_meth = {}

    if data != []:
        dataset_l = data
    else:
        dataset_l = dataset_names.keys()

    if method != []:
        method_l = method
    else:
        method_l = methods

    for meth in method_l:
        std_meth[meth] = 0

        for dataset_name in dataset_l:

            results_meth_level = pd.read_csv(f'eval/gen_data/{dataset_name}/{meth}/results_{meth}.csv', index_col=0)
            std_meth_level = results_meth_level.iloc[:-1,:].std()
            std_meth[meth] += std_meth_level

    std_meth_df = pd.DataFrame.from_dict(std_meth, orient='index')
    std_meth_avg = std_meth_df / len(dataset_l)
    std_meth_avg = std_meth_avg.round(decimals=2)
    std_meth_avg.to_csv(f'eval/std_results.csv')

def calc_std_diff(data, method):

    meth_level_all = {}

    if data != []:
        dataset_l = data
    else:
        dataset_l = dataset_names.keys()

    if method != []:
        method_l = method
    else:
        method_l = methods

    for meth in method_l:
        meth_level_all[meth] = []

        for dataset_name in dataset_l:

            results_meth_level = pd.read_csv(f'eval/gen_data/{dataset_name}/{meth}/results_{meth}.csv', index_col=0)
            meth_level = results_meth_level.iloc[:-1,:]
            meth_level_all[meth].append(meth_level)

        meth_level_all[meth] = pd.concat(meth_level_all[meth])

    for meth_a in method_l:
        for meth_b in method_l:
            diff_1 = (meth_level_all[meth_a]['ups'].loc[0] - meth_level_all[meth_b]['ups'].loc[0])
            diff_2 = (meth_level_all[meth_a]['ups'].loc[1] - meth_level_all[meth_b]['ups'].loc[1])
            t_stat, p_value = stats.ttest_ind(diff_1.abs(), diff_2.abs())
            t_stat, p_value = stats.ttest_ind(diff_1.abs(), diff_2.abs())
            print(f"\nMethod pair {meth_a} - {meth_b}\n", t_stat, p_value) # zwischen allen Datenpaaren durchschnitt des p values berechnen? Dann mit FRG besrpechen

    #std_meth_avg.to_csv(f'eval/std_results.csv')

def corr_ups_us(data, method):
    df_coll = []

    if data != []:
        dataset_l = data
    else:
        dataset_l = dataset_names.keys()


    for dataset_name in dataset_l:

        results_data_level = pd.read_csv(f'eval/gen_data/{dataset_name}/results_{dataset_name}.csv', index_col=0)
        df_coll.append(results_data_level[['us', 'pps']])

    df = pd.concat(df_coll)
    print(df)
    print(stats.pearsonr(df['us'],df['pps']))

    # std_meth_df = pd.DataFrame.from_dict(std_meth, orient='index')
    # std_meth_avg = std_meth_df / len(dataset_l)
    # std_meth_avg = std_meth_avg.round(decimals=2)
    # std_meth_avg.to_csv(f'eval/std_results.csv')

def corr_task_size(data, method):

    ups_values = {}
    if data != []:
        dataset_l = data
    else:
        dataset_l = dataset_names.keys()

    if method != []:
        method_l = method
    else:
        method_l = methods

    dataset_sizes = {}
    for dataset_name in dataset_l:
        dataset, class_var, cat_feat_names, num_feat_names = load_data(dataset_name)
        train_data = pd.read_csv(f'eval/train_data/{dataset_name}/spl_1.csv', index_col=0)
        dataset_sizes[len(list(train_data[class_var].unique()))] = dataset_name
    dataset_sizes_sorted = {key: dataset_sizes[key] for key in sorted(dataset_sizes)}
    dataset_l_sorted = dataset_sizes_sorted.values()

    for meth in method_l:
        ups_values[meth] = []
        for dataset_name in dataset_l_sorted:
            print(dataset_name)
            results_data_level = pd.read_csv(f'eval/gen_data/{dataset_name}/results_{dataset_name}.csv', index_col=0)
            ups_values[meth].append(float(results_data_level['ups'][results_data_level['method']==meth].values))

    sample_sizes = list(range(1, 17))  # 17 verschiedene Sample Sizes
    plt.figure(figsize=(12, 8))

    for meth in method_l:
        #if meth == 'great':
        plt.plot(sample_sizes, ups_values[meth], label=meth, marker='o')  # Jede Methode als Linie

    # Diagramm-Details
    plt.title("Corr ups - task_size", fontsize=16)
    plt.xlabel("# classes", fontsize=11)
    plt.ylabel("UPS", fontsize=11)
    #plt.legend(title="Methoden", fontsize=10, title_fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Diagramm anzeigen
    plt.show()
    plt.savefig("corr_task")

def corr_feat_size(data, method):
    ups_values = {}
    if data != []:
        dataset_l = data
    else:
        dataset_l = dataset_names.keys()

    if method != []:
        method_l = method
    else:
        method_l = methods

    dataset_sizes = {}
    for dataset_name in dataset_l:
        train_data = pd.read_csv(f'eval/train_data/{dataset_name}/spl_1.csv', index_col=0)
        dataset_sizes[train_data.shape[1]] = dataset_name
    dataset_sizes_sorted = {key: dataset_sizes[key] for key in sorted(dataset_sizes)}
    dataset_l_sorted = dataset_sizes_sorted.values()

    for meth in method_l:
        ups_values[meth] = []
        for dataset_name in dataset_l_sorted:
            print(dataset_name)
            results_data_level = pd.read_csv(f'eval/gen_data/{dataset_name}/results_{dataset_name}.csv', index_col=0)
            ups_values[meth].append(float(results_data_level['ups'][results_data_level['method']==meth].values))

    sample_sizes = list(range(1, 17))  # 17 verschiedene Sample Sizes
    plt.figure(figsize=(12, 8))

    for meth in method_l:
        #if meth == 'great':
        plt.plot(sample_sizes, ups_values[meth], label=meth, marker='o')  # Jede Methode als Linie

    # Diagramm-Details
    plt.title("Corr ups - feat", fontsize=16)
    plt.xlabel("Feat Size", fontsize=11)
    plt.ylabel("UPS", fontsize=11)
    #plt.legend(title="Methoden", fontsize=10, title_fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Diagramm anzeigen
    plt.show()
    plt.savefig("corr_feat")

def corr_smpl_size(data, method):
    ups_values = {}
    if data != []:
        dataset_l = data
    else:
        dataset_l = dataset_names.keys()

    if method != []:
        method_l = method
    else:
        method_l = methods

    dataset_sizes = {}
    for dataset_name in dataset_l:
        train_data = pd.read_csv(f'eval/train_data/{dataset_name}/spl_1.csv', index_col=0)
        dataset_sizes[train_data.shape[0]] = dataset_name
    dataset_sizes_sorted = {key: dataset_sizes[key] for key in sorted(dataset_sizes)}
    dataset_l_sorted = dataset_sizes_sorted.values()

    for meth in method_l:
        ups_values[meth] = []
        for dataset_name in dataset_l_sorted:
            results_data_level = pd.read_csv(f'eval/gen_data/{dataset_name}/results_{dataset_name}.csv', index_col=0)
            ups_values[meth].append(float(results_data_level['ups'][results_data_level['method']==meth].values))

    sample_sizes = list(range(1, 17))  # 17 verschiedene Sample Sizes
    plt.figure(figsize=(12, 8))

    for meth in method_l:
        #if meth == 'great':
        plt.plot(sample_sizes, ups_values[meth], label=meth, marker='o')  # Jede Methode als Linie

    # Diagramm-Details
    plt.title("corr ups - smpl_size", fontsize=16)
    plt.xlabel("Sample Size", fontsize=11)
    plt.ylabel("UPS", fontsize=11)
    #plt.legend(title="Methoden", fontsize=10, title_fontsize=12, loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Diagramm anzeigen
    plt.show()
    plt.savefig("corr_smpl")

def corr_feat_type(data, method):

    if data != []:
        dataset_l = data
    else:
        dataset_l = dataset_names.keys()

    datasets_type = {}
    for dataset_name in dataset_l:
        dataset, class_var, cat_feat_names, num_feat_names = load_data(dataset_name)
        if len(cat_feat_names) == 0:
            datasets_type['num'] = dataset_name
        elif len(num_feat_names) == 0:
            datasets_type['cat'] = dataset_name
        else:
            if (len(cat_feat_names) / len(num_feat_names)) < 1:
                datasets_type['num'] = dataset_name
            else:
                datasets_type['cat'] = dataset_name

    ups_dict = {}
    for t in datasets_type.keys():
        ups_dict[t] = {}
        for dataset_name in datasets_type[t]:
            results_data_level = pd.read_csv(f'eval/gen_data/{dataset_name}/results_{dataset_name}.csv', index_col=0)
            cols = results_data_level['method'].values.tolist()
            ups_dict[t][dataset_name] = results_data_level['ups']
        ups_df = pd.DataFrame.from_dict(ups_dict[t], orient='index')
        ups_df.columns = cols
        ups_df['dataset'] = ups_dict[t].keys()
        ups_df.to_csv(f'eval/ups_{t}.csv')

def anova(data, method):
    import pandas as pd
    import numpy as np
    from scipy.stats import f_oneway

    if data != []:
        dataset_l = data
    else:
        dataset_l = dataset_names.keys()

    ups_dict = {}
    for dataset_name in dataset_l:
        results_data_level = pd.read_csv(f'eval/gen_data/{dataset_name}/results_{dataset_name}.csv', index_col=0)
        cols = results_data_level['method'].values.tolist()
        ups_dict[dataset_name] = results_data_level['ups']

    ups_df = pd.DataFrame.from_dict(ups_dict, orient='index')
    ups_df.columns = cols
    ups_df['dataset'] = ups_dict.keys()
    print(ups_df)


    # ANOVA
    f_stat, p_value = f_oneway(ups_df['tvae'], ups_df['gausscop'], ups_df['ctgan'], ups_df['arf'], ups_df['nflow'], ups_df['knnmtd'], ups_df['mcgen'], ups_df['corgan'], ups_df['smote'], ups_df['priv_bayes'], ups_df['cart'])
    print(f"F-Wert: {f_stat}, p-Wert: {p_value}")
    #
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Daten in Long-Format bringen
    long_df = ups_df.melt(id_vars='dataset', var_name='method', value_name='result')

    # Tukey-Test
    tukey = pairwise_tukeyhsd(endog=long_df['result'], groups=long_df['method'], alpha=0.05)
    print(tukey)



# start of command line call and loading of arguments
if __name__ == "__main__":

    # defines parsers
    parser = argparse.ArgumentParser()

    # arguments for data generation
    parser.add_argument('--type', type=str, required=True, help='calculates the standard deviations for the performance of each synthesizers over all splits of each dataset => averages over all datasets are returned in csv')
    #parser.add_argument('corr_smpl_size', type=bool, default=False, help='calculates the standard deviations for the performance of each synthesizers over all splits of each dataset => averages over all datasets are returned in csv')
    #parser.add_argument('corr', type=bool, default=False, help='calculates the standard deviations for the performance of each synthesizers over all splits of each dataset => averages over all datasets are returned in csv')
    #parser.add_argument('calc_std', type=bool, default=False, help='calculates the standard deviations for the performance of each synthesizers over all splits of each dataset => averages over all datasets are returned in csv')
    #parser.add_argument('calc_std_diff', type=bool, default=False, help='calculates the standard deviations for the performance of each synthesizers over all splits of each dataset => averages over all datasets are returned in csv')
    #parser.add_argument('anova', type=bool, default=False, help='calculates anova to judge statistical significance of avg ups differences between the methods over all datasets')
    parser.add_argument('--data', type=str, required=False, nargs='+', default=[],
                            help="Select dataset name, default iteratively takes all datasets")
    parser.add_argument('--n_spl', type=int, required=False, default='10', help="Choose number of data splits")
    parser.add_argument('--method', type=str, required=False, nargs='+', default=[],
                            help="Select method by name")  # tvae, gausscop, ctgan / arf / pzflow / knnmtd / mcgen / corgan / ensgen / genary / smote / priv_bayes, cart

    args = parser.parse_args()

    if args.type=='corr_smpl_size':
        corr_smpl_size(args.data, args.method)
    if args.type=='corr_feat_size':
        corr_feat_size(args.data, args.method)
    # if args.calc_std_diff:
    #     calc_std_diff(args.data, args.method)
    # if args.calc_std:
    #     calc_std(args.data, args.method)
    #if args.anova:
     #   anova(args.data, args.method)
    # if args.corr:
    #   anova(args.data, args.method)
