from unittest.mock import inplace

import pandas as pd
import argparse
import scipy.stats as stats
import matplotlib.pyplot as plt
from importlib import import_module, reload
preprocessing = import_module('data.preprocessing')
import numpy as np
import seaborn
#from pkg_resources import require
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
    std_meth_avg.to_csv(f'eval/std_results_spl.csv')

    results_summary = pd.read_csv(f'eval/results_summary_std.csv', index_col=0)
    print("AVG Std across all splits", std_meth_avg.mean())
    print("AVG Std across all datasets", results_summary.iloc[:,1:].mean())

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

def corr_us_pps(data, method):
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

    seaborn.regplot(x='us', y='pps', data=df)
    plt.title('utility-privacy trade-off trend line')
    plt.savefig('eval/results/us-pps-corr.png', dpi=200)
    plt.show()


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

    dataset_sizes = {'=2': [], '>2': []}
    for dataset_name in dataset_l:
        dataset, class_var, cat_feat_names, num_feat_names = load_data(dataset_name)
        train_data = pd.read_csv(f'eval/train_data/{dataset_name}/spl_1.csv', index_col=0)
        class_var = type(train_data.columns[1])(class_var)
        #dataset_sizes[dataset_name] = len(list(train_data[class_var].unique()))
        num_cl = len(list(train_data[class_var].unique()))
        if num_cl == 2:
            dataset_sizes['=2'].append(dataset_name)
        else:
            dataset_sizes['>2'].append(dataset_name)
    #dataset_sizes_sorted = dict(sorted(dataset_sizes.items(), key=lambda item: item[1]))
    #dataset_l_sorted = dataset_sizes_sorted.keys()

    for size in dataset_sizes.keys():
        ups_values[size] = 0
        for dataset_name in dataset_sizes[size]:
            results_data_level = pd.read_csv(f'eval/gen_data/{dataset_name}/results_{dataset_name}.csv', index_col=0)
            ups_values[size] += results_data_level.iloc[:, 1:]
        ups_values[size] /= len(dataset_sizes[size])
        ups_values[size]['method'] = results_data_level['method']
        ups_values[size] = ups_values[size][results_data_level.columns]
        ups_values[size].to_csv(f'eval/results/corr_task_{size}.csv')


def corr_size(data, mode='smpl'):
    ups_values = {'small': 0, 'large': 0}
    if data != []:
        dataset_l = data
    else:
        dataset_l = dataset_names.keys()

    dataset_sizes = {}
    for dataset_name in dataset_l:
        train_data = pd.read_csv(f'eval/train_data/{dataset_name}/spl_1.csv', index_col=0)
        dataset_sizes[dataset_name] = train_data.shape[int(mode=='feat')]
    dataset_sizes_sorted = dict(sorted(dataset_sizes.items(), key=lambda item: item[1]))
    dataset_l_sorted = dataset_sizes_sorted.keys()

    c = {'small': 0, 'large': 0}
    z = 0
    for dataset_name in dataset_l_sorted:
        if z < 9:
            k = 'small'
            c[k] += 1
        else:
            k = 'large'
            c[k] += 1
        results_data_level = pd.read_csv(f'eval/gen_data/{dataset_name}/results_{dataset_name}.csv', index_col=0)
        ups_values[k] += results_data_level.iloc[:, 1:]
        z += 1
    for k in c.keys():
        ups_values[k] /= c[k]
        ups_values[k]['method'] = results_data_level['method']
        ups_values[k] = ups_values[k][results_data_level.columns]
        ups_values[k].to_csv(f'eval/results/corr_{mode}_{k}.csv')

    # sample_sizes = list(range(0, 17))  # 17 verschiedene Sample Sizes
    # plt.figure(figsize=(12, 8))
    #
    # for meth in method_l:
    #     plt.plot(sample_sizes, ups_values[meth], label=meth, marker='o')  # Jede Methode als Linie

    # # Diagramm-Details
    # plt.xticks(ticks=sample_sizes, labels=dataset_sizes_sorted.values())
    # plt.title("Correlation between ups and feature size", fontsize=16)
    # plt.xlabel("feature size", fontsize=11)
    # plt.ylabel("ups", fontsize=11)
    # #plt.legend(title="Methoden", fontsize=10, title_fontsize=12, loc='best')
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    #
    # # Diagramm anzeigen
    # plt.show()
    # plt.savefig("corr_feat")

def corr_feat_type(data, method):

    if data != []:
        dataset_l = data
    else:
        dataset_l = dataset_names.keys()

    datasets_type = {'num': [], 'cat': []}
    for dataset_name in dataset_l:
        dataset, class_var, cat_feat_names, num_feat_names = load_data(dataset_name)
        if len(cat_feat_names) == 0:
            datasets_type['num'].append(dataset_name)
        elif len(num_feat_names) == 0:
            datasets_type['cat'].append(dataset_name)
        else:
            if (len(cat_feat_names) / len(num_feat_names)) < 1:
                datasets_type['num'].append(dataset_name)
            else:
                datasets_type['cat'].append(dataset_name)

    ups_dict = {}
    summ_df = pd.DataFrame()
    for t in datasets_type.keys():
        ups_dict[t] = 0
        for dataset_name in datasets_type[t]:
            results_data_level = pd.read_csv(f'eval/gen_data/{dataset_name}/results_{dataset_name}.csv', index_col=0)
            cols = results_data_level['method'].values.tolist()
            ups_dict[t] += results_data_level.iloc[:, 1:]
        #ups_df = pd.DataFrame.from_dict(ups_dict[t], orient='index')
        ups_dict[t] /= len(datasets_type[t])
        ups_dict[t]['method'] = cols
        ups_dict[t] = ups_dict[t][results_data_level.columns]
        ups_dict[t].to_csv(f'eval/results/corr_{t}.csv')

def check_data(data, method, n_spl):
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
        for i in range(n_spl):
            i +=1
            train_data = pd.read_csv(f'eval/train_data/{dataset_name}/spl_{i}.csv', index_col=0)
            test_data = pd.read_csv(f'eval/test_data/{dataset_name}/spl_{i}.csv', index_col=0)
            for meth in method_l:
                train_data_meth = pd.read_csv(f'eval/train_data/{dataset_name}/{meth}/spl_{i}.csv', index_col=0)
                test_data_meth = pd.read_csv(f'eval/test_data/{dataset_name}/{meth}/spl_{i}.csv', index_col=0)
                try:
                    assert (train_data == train_data_meth).all().all()
                    assert train_data.equals(train_data_meth)
                    assert (test_data == test_data_meth).all().all()
                    assert test_data.equals(test_data_meth)
                    #assert train_data_meth.eq(test_data_meth).any().any()==False
                    #assert train_data.eq(test_data).any().any()==False
                except:
                    print(f"{dataset_name}_{meth}_spl_{i}")

def summary():
    corr_smpl_small = pd.read_csv(f'eval/results/corr_smpl_small.csv', index_col=0)
    corr_smpl_large = pd.read_csv(f'eval/results/corr_smpl_large.csv', index_col=0)
    corr_feat_small = pd.read_csv(f'eval/results/corr_feat_small.csv', index_col=0)
    corr_feat_large = pd.read_csv(f'eval/results/corr_feat_large.csv', index_col=0)
    corr_num = pd.read_csv(f'eval/results/corr_num.csv', index_col=0)
    corr_cat = pd.read_csv(f'eval/results/corr_cat.csv', index_col=0)
    corr_task_small = pd.read_csv(f'eval/results/corr_task_=2.csv', index_col=0)
    corr_task_large = pd.read_csv(f'eval/results/corr_task_>2.csv', index_col=0)
    for metric in ['ups', 'us', 'pps']:
        summ_df = pd.DataFrame()
        summ_df['method'] = corr_smpl_small['method']
        summ_df['smpl<'] = corr_smpl_small[metric]
        summ_df['smpl>'] = corr_smpl_large[metric]
        summ_df['feat<'] = corr_feat_small[metric]
        summ_df['feat>'] = corr_feat_large[metric]
        summ_df['num'] = corr_num[metric]
        summ_df['cat'] = corr_cat[metric]
        summ_df['class<'] = corr_task_small[metric]
        summ_df['class>'] = corr_task_large[metric]
        summ_df_avg = summ_df.iloc[:,1:].mean()
        summ_df_avg.T['method'] = 'avg'
        summ_df_avg = summ_df_avg[summ_df.columns]
        summ_df = pd.concat([summ_df, summ_df_avg.to_frame().T])
        summ_df.to_csv(f'eval/results/corr_summary_{metric}.csv')
        if metric == 'ups':
            df = summ_df.iloc[:-1, :].copy()

    # bar graph of differences in each data characteristic
    summ_df_diff = pd.DataFrame({'smpl_diff': abs(df["smpl<"] - df["smpl>"]),
                                 'feat_diff': abs(df["feat<"] - df["feat>"]),
                                 'num_cat_diff': abs(df["num"] - df["cat"]),
                                 'class_diff': abs(df["class<"] - df["class>"])
                                 })

    syn_rob = (1-summ_df_diff.mean(axis=1)) * 100
    syn_rob_df = pd.DataFrame({'synthesizer': df['method'], 'robustness': syn_rob})
    syn_rob_df.sort_values(by='robustness', ascending=False, inplace=True)
    bar_values = syn_rob_df['robustness']
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(syn_rob_df.shape[0])
    bar_width = 0.9
    #colors = ["blue", "orange", "green", "red"]
    #labels = ["smpl<>", "feat<>", "num_cat", "class<>"]

    #for i, col in enumerate(bar_values.columns):
        #ax.bar(x + i * bar_width, bar_values[col], width=bar_width, color=colors[i], label=labels[i])
    ax.bar(x + 1.5*bar_width, bar_values, width=bar_width, color='black')

    ax.set_xlabel("synthesizers")
    ax.set_ylabel("robustness in %")
    #ax.set_title("Robustness of synthesizers against different data characteristics")
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(syn_rob_df['synthesizer'], rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.rcParams.update({'font.size': 11})
    plt.ylim(88, 100)
    plt.savefig("eval/results/syn_robustness.png", dpi=200)
    plt.show()


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
        corr_size(args.data, 'smpl')
    if args.type=='corr_feat_size':
        corr_size(args.data, 'feat')
    if args.type == 'check_data':
        check_data(args.data, args.method, args.n_spl)
    if args.type == 'calc_std':
        calc_std(args.data, args.method)
    if args.type == 'corr_feat_type':
        corr_feat_type(args.data, args.method)
    if args.type == 'corr_us_pps':
        corr_us_pps(args.data, args.method)
    if args.type == 'corr_task_size':
        corr_task_size(args.data, args.method)
    if args.type == 'summary':
        summary()