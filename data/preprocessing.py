import numpy as np
import pandas as pd
import csv
import re
import random

# selection of datasets and their os paths
def load_data(dataset_name):
    data_paths = {
        'arrhythmia': 'Arrhythmia/arrhythmia.data.txt',
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
        'eye': 'EEG Eye State/EEG Eye State.arff.txt'
    }

    
    print("Dataset:", dataset_name)
    print("---------------------------")

    # current path because this module is imported and so relative paths change in the different location
    import os
    orig_path = os.path.dirname(os.path.abspath(__file__)) + '/'

    # select dataset (name should be defined via cmd argument)

    #dataset_name = 'eye'
    data_path = data_paths[dataset_name]
    data_path = orig_path + data_path
    
    # read meta information about available datasets
    meta_path = orig_path + 'datasets_meta.xlsx'
    dataset_meta = pd.read_excel(meta_path)
    dataset_meta.set_index('Name', inplace=True) # set names of datasets as index (for access by index)
    num_cl = dataset_meta.at[dataset_name, '#Classes'] # .at accesses exactly one cell in the dataframe

    # to read data from a text file and return a pandas dataframe
    def read_from_text(data_path):
        dataset = []
        with open(data_path, 'r') as f:
            #dataset = f.readlines()
            for line in f:
                line = line.replace('\t', '').replace('\n', '')
                line_list = line.split(',')
                line_len = len(line_list)
                try:
                    if line_len != line_len_prev:
                        print("Line length changes!\n", line_list)
                except:
                    line_len_prev = len(line_list)
                if '\t' in line_list:
                    del line_list[line_list.index('\t')]
                if '\n' in line_list:
                    del line_list[line_list.index('\n')]
                for i in range(len(line_list)):
                    if bool(re.search('^[0-9]+$', line_list[i])) == True:
                        line_list[i] = int(line_list[i])
                        continue
                    if bool(re.search('^\d*[\.]\d+$', line_list[i])) == True:
                        line_list[i] = float(line_list[i])

                dataset.append(line_list)
        return dataset

    # %%
    # read data depending on file format

    if ".csv" in data_path:
        dataset = pd.read_csv(data_path)
    if ".xls" in data_path:
        dataset = pd.read_excel(data_path)
    elif ".txt" in data_path:
        dataset = read_from_text(data_path)
    df = pd.DataFrame(dataset)

    # remove index column if imported from file
    if list(df.iloc[:,0]) == list(range(0,len(df.iloc[:,0]))):
        df.drop(columns=df.iloc[:,0].name, inplace=True)

    # remove 'id' column is exists >> unique values are useless in machine learning
    if 'id' in df.columns:
        df.drop(columns='id', inplace=True)
    if 'ID' in df.columns:
        df.drop(columns='ID', inplace=True)
    
    # checks for missing values and replaces them with common feature values

    if (df=='?').any().any() == True:
        df.replace(to_replace='?', value=np.NaN, inplace=True)

    if df.isnull().any().any() == True:
        print("Missing values 'NaN':", df.isnull().sum().sum())

        for feat in df:

            if df[feat].isnull().any() == True:

                ## for continuous feature values
                if df[feat].dtype == 'float':
                    df[feat].fillna(df[feat].median(), inplace = True)

                ## for discrete feature values
                else:
                    dic_weighs = {}
                    for key in df[feat].value_counts().keys():
                        dic_weighs[key] = df[feat].value_counts()[key] / df[feat].value_counts().sum()

                    idx_keys = list(df[df[feat].isnull()].index.values)
                    rand_nom = random.choices(list(dic_weighs.keys()), weights=list(dic_weighs.values()), k=len(idx_keys))
                    dic_rpl = {idx_keys[i]: rand_nom[i] for i in range(len(idx_keys))}
                    df[feat].fillna(dic_rpl, inplace = True)

        print("Missing values:", df.isnull().any()[0])

    # %%
    # check if number of class labels matches any of the columns (known from meta information)

    ## checks if last column contains class labels (often the case)
    if len(df.iloc[:,-1].unique()) == num_cl:
        print("last line with name {} is class column!".format(df.iloc[:,-1].name))
        flag_cl_last = True
        cl_col = df.iloc[:,-1].name

    ## checks all datasets columns > if several columns are printed => error!
    else:
        print("last columns is NOT class column - it has {} instead of {} labels".format(len(df.iloc[:,-1].unique()), num_cl))
        c = 0
        for f in range(df.shape[1]):
            if len(df.iloc[:,f].unique()) == num_cl:
                print("Potential class column:", df.iloc[:,f].name)
                print(df.iloc[:,f].unique())
                cl_col = df.iloc[:,f].name
                c += 1
                if c > 1:
                    raise Exception("Multiple potential class columns!") 
                
    
    #  remove features with only one unique 1 value -> contains no useful information
    rm_feat = 0 # tracks how many features will be removed due to singular values
    for feat in df:
        min_vals = []
        if len(df[feat].unique()) == 1:
            df.drop(columns=feat, inplace=True)
            rm_feat += 1
            continue

    print("{} features removed because only one value".format(rm_feat))



    # compare information extracted from dataset with meta information
    contFeat = 0
    discFeat = 0

    if num_cl != len(df[cl_col].unique()):
        print("Num classes do not match! {} vs {}".format(num_cl, len(df[cl_col].unique())))
        num_cl = len(df[cl_col].unique())

    if dataset_meta.at[dataset_name,'#Records'] != df.shape[0]:
        print("Num records do not match! {} vs {}".format(dataset_meta.at[dataset_name,'#Records'], df.shape[0]))

    if dataset_meta.at[dataset_name,'#Features'] != df.shape[1]-1:
        print("Num features do not match! {} vs {}".format(dataset_meta.at[dataset_name,'#Features'], df.shape[1]-1))

    # check number of continuous/discrete features (identification important for later synthesis)
    disc_feat_names = []
    cont_feat_names = []
    for feat in df:

        # dont include class label in analysis
        if feat == cl_col:
            continue

        # check if feature type is string => discrete
        if df[feat].dtype == 'object':
            discFeat += 1
            disc_feat_names.append(feat)
            continue

        # check if feature is ordinal
        sorted_feat = sorted(list(df[feat].unique()))
        for i in range(len(sorted_feat)):
            if i == (len(sorted_feat)-1):
                discFeat += 1
                disc_feat_names.append(feat)
                flag_cont = False
                break
            if (sorted_feat[i] - sorted_feat[i+1]) != -1:
                contFeat += 1
                cont_feat_names.append(feat)
                flag_cont = True
                break
        
        # make sure there are no categories not numbered in order and therefore not detected in the check before > 10% or less unique values is suspicious
        if (flag_cont == True) and ( (len(sorted_feat) / len(list(df[feat]))) < 0.11):
            #print("Feature", feat, "less 10:", len(sorted_feat))
            discFeat += 1
            disc_feat_names.append(feat)
            contFeat -= 1
            cont_feat_names.remove(feat)


    # to remove classes with only one records (is not represenative)
    def check_cl_size(df, cl_col):
        for cl in df[cl_col].unique():
            cl_df = df[df[cl_col]==cl]
            if cl_df.shape[0] < 2:
                df.drop(index=cl_df.index, inplace=True)

        return df

    df = check_cl_size(df, cl_col)

    print("Number of records:", df.shape[0])
    print("Number of features:", df.shape[1])
    print("Num continuous features {} vs {}:".format(dataset_meta.at[dataset_name,'#ContFeat'],contFeat))
    print("Num discrete features {} vs {}:".format(dataset_meta.at[dataset_name,'#DiscFeat'],discFeat))
    print("Class variable:", cl_col, "(", num_cl, "classes)")

    return df, cl_col, disc_feat_names, cont_feat_names
