import numpy as np
import pandas as pd
import random
from importlib import reload, import_module
import statistics
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
import os, glob
from tqdm import tqdm
import sys
#sys.path.append("/Users/juho/Promotion/FakeData/Paper/IEEE/data") # replace with import_module?
#import preprocessing
import argparse
preprocessing = import_module('data.preprocessing') # preprocessing module
from eval.pps import PPS
from collections import Counter

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

# available methods
methods = ['tvae', 'gausscop', 'ctgan', 'arf', 'nflow', 'knnmtd', 'mcgen', 'corgan', 'ensgen', 'genary', 'smote',
           'priv_bayes', 'cart', 'great', 'tabula']

# ## Configuration and Dataset Prepraration

def load_data(dataset_name):
    reload(preprocessing)
    dataset, class_var, cat_feat_names, num_feat_names = preprocessing.load_data(dataset_name)
    
    return dataset, class_var, cat_feat_names, num_feat_names

def prepData(dataset, class_var):

    data = dataset.copy()
    dataset_feat = data.drop(labels=class_var, axis=1)
    features = list(dataset_feat.columns)

    # encode string values
    # category_mapping = dict(enumerate(data[feat].astype('category').cat.categories)) to save the mappings > could be reassigned after synthesis
    # data[feat] = data[feat].astype('category').cat.codes
    for feat in data:
        if data[feat].dtype == 'object':
            data[feat] = data[feat].astype('category').cat.codes

    # converts class variable to int (float may cause problems when fitting sklearn models)
    if data[class_var].dtype == 'float':
            data[class_var] = data[class_var].astype('int')

    # data split into train/test
    test_set = data.groupby(class_var, group_keys=False).apply(lambda x: x.sample(frac=0.5, random_state=random.randrange(1000000))) # each class equally sampled (to avoid different classes in train/test)
    data.drop(index=test_set.index, inplace=True)
    train_set = data.copy()

    return train_set, test_set
    

# ### Evaluation Functions

# function to initialize classifier
def clfType(model):
    if model == 'kNN':
        clf = KNeighborsClassifier()
    if model == 'NB':
        clf = GaussianNB()
    if model == 'LG':
        clf = LogisticRegression()
    if model == 'DT':
        clf = DecisionTreeClassifier(random_state=567)
    if model == 'RF':
        clf = RandomForestClassifier(max_depth=3, random_state=567)

    return clf

# function to test a classifier and print the f1-score
def testClf(X_train, y_train, X_test, y_test, descr, model, printScore):
    
    # Training and test of model
    clf = clfType(model)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    if printScore == True:
        print(descr, ": ", f1)
    return f1, preds

# converts continuous values to discete values if feature was originally discrete 
def conv_to_disc(gen_data, cat_feat_names):
    cols = list(gen_data.columns)
    for col in cols:
        if col in cat_feat_names:
            gen_data[col] = round(gen_data[col])
    return gen_data

# for some methods needed if it can't deal with a low number of records per class
def check_cl_size(train_set, test_set, class_var, feat):
    for cl in train_set[class_var].unique():
        cl_data_train = train_set[feat][train_set[class_var]==cl]
        cl_data_test = test_set[feat][test_set[class_var]==cl]
        if cl_data_train.shape[0] < 5 or cl_data_test.shape[0] < 5:
            train_set.drop(index=cl_data_train.index, inplace=True)
            test_set.drop(index=cl_data_test.index, inplace=True)
            print(f"class {cl} removed!")
    return train_set, test_set



# data synthesis process
def gen(data, n_spl, method, smpl_frac, test):

    for dataset_name, dataset_path in dataset_names.items():
        if args.data != 'all':
            dataset_name, dataset_path = args.data, dataset_names[args.data]

        print("\n")

        dataset, class_var, cat_feat_names, num_feat_names = load_data(dataset_name)
        if args.test:
            if dataset.shape[0] > 500:
                i=0
                while i < 2:
                    dataset = dataset.sample(n=500, random_state=22)
                    i = len(dataset[class_var].unique())
        cols = list(dataset.columns)
        cols.remove(class_var)
        feat = cols

        # store f1 performances of synthetic/real data models
        f1_real_all = []
        f1_syn_all = []

        # experiment loop
        for i in tqdm(range(n_spl)):
            i += 1

            # prepares training / test data
            train_set, test_set = prepData(dataset, class_var)
            train_set, test_set = check_cl_size(train_set, test_set, class_var, feat) # removes classes with sample size < 5

            if method != 'all':
                method_l = [method]
            else:
                method_l = methods

            for meth in tqdm(method_l):

                print(f"\n METHOD {meth} ...")

                # run data synthesis script
                if meth in ('gausscop', 'tvae', 'ctgan'):
                    sdv = import_module('methods.sdv.sdv')
                    gen_data = sdv.gen(train_set.copy(), smpl_frac, meth, cat_feat_names.copy(), num_feat_names.copy(), class_var)
                    gen_data.columns = list(test_set.columns)
                    train_set.columns = list(test_set.columns)
                if meth == 'arf':
                    arf = import_module('methods.ARF.arf')
                    gen_data = arf.gen(train_set.copy(), feat.copy(), args.smpl_frac, class_var)
                if meth == 'nflow':
                    pzflow = import_module('methods.PZFlow.pzflow')
                    gen_data = pzflow.gen(train_set.copy(), feat.copy(), args.smpl_frac, class_var)
                    gen_data = conv_to_disc(gen_data, cat_feat_names)
                if meth == 'knnmtd':
                    knnmtd = import_module('methods.knnMTD.knnmtd')
                    gen_data = knnmtd.gen(train_set.copy(), smpl_frac, class_var)
                    gen_data = conv_to_disc(gen_data.copy(), cat_feat_names.copy())
                if meth == 'mcgen':
                    mcgen = import_module('methods.MCGEN.mcgen')
                    gen_data = mcgen.gen(train_set.copy(), smpl_frac, class_var, dataset_name)
                    gen_data = conv_to_disc(gen_data, cat_feat_names)
                if meth == 'corgan':
                    corgan = import_module('methods.CorGAN.corgan')
                    gen_data = corgan.gen(train_set.copy(), smpl_frac, class_var, feat.copy(), cat_feat_names.copy())
                    gen_data = conv_to_disc(gen_data, cat_feat_names)
                if meth == 'ensgen':
                    ensgen = import_module('methods.ensemble.ensgen')
                    gen_data_ens = ensgen.gen(train_set.copy(), feat.copy(), class_var, cat_feat_names.copy(), num_feat_names.copy(), smpl_frac_hist=0.5, n_bins=30, n_ens=30) # orig: smpl_frac_hist=0.5, n_bins=30, n_ens=30
                    gen_data = pd.concat(gen_data_ens, axis=0) # appends all synthetic datasets (for saving purposes - need to be seperated accordingly in evaluation)
                if meth == 'smote':
                    smote = import_module('methods.SMOTE.smote')
                    gen_data = smote.gen(train_set.copy(), feat.copy(), class_var, smpl_frac, num_nn=3)
                if meth == 'genary': # method needs to be adjusted so that algorithm tests against all models
                    # sklearns kNN in this method causes problems when dealing with large datasets
                    genary = import_module('methods.evolutionary.genary')
                    selec = genary.gen(train_set.copy(), feat.copy(), class_var, cat_feat_names.copy(), num_feat_names.copy(), smpl_frac, e_steps=100, n_cand=100, n_selec=50, crov_frac_pop=0.95, crov_frac_indv=0.5, mut_frac_pop=0.75, mut_frac_indv=0.03)
                    gen_data = selec[0]
                if meth == 'great': # sampling only works using gpu in standard code >> can be changed, though
                    great = import_module('methods.GReaT.great') # https://github.com/kathrinse/be_great/tree/main
                    gen_data = great.gen(train_set.copy(), smpl_frac, llm_id='distilgpt2') # distil-gpt2 / gpt2-medium
                if meth == 'tabula': # sampling only works using gpu in standard code >> can be changed, though
                    tabula = import_module('methods.TabuLa.Tabula-main.tabula_gen') # https://github.com/zhao-zilong/tabula
                    gen_data = tabula.gen(train_set.copy(), cat_feat_names.copy(), smpl_frac)
                if meth in ('priv_bayes', 'cart'): # synthpop package loads the parametric approach by default (was replaced with DT classifier class)
                    dpart = import_module('methods.dpart.dpart_gen')
                    gen_data = dpart.gen(train_set.copy(), feat.copy(), class_var, meth, smpl_frac, eps=0.5)

                # optional: remap original feature values
                #data[feat] = data[feat].map(category_mapping)

                # save the train, test and generated data for each method, dataset and split
                os.makedirs(f'eval/gen_data/{dataset_name}/{meth}', exist_ok=True)
                os.makedirs(f'eval/train_data/{dataset_name}/{meth}', exist_ok=True)
                os.makedirs(f'eval/test_data/{dataset_name}/{meth}', exist_ok=True)
                gen_data.to_csv(f'eval/gen_data/{dataset_name}/{meth}/spl_{i}.csv')
                train_set.to_csv(f'eval/train_data/{dataset_name}/{meth}/spl_{i}.csv')
                test_set.to_csv(f'eval/test_data/{dataset_name}/{meth}/spl_{i}.csv')

        if args.data != 'all':
            break


def ens_pred(gen_data, train_set, test_set, feat, class_var, model):
    ensemblePreds = []
    for i in range(int(gen_data.shape[0] / train_set.shape[0])):
        data = gen_data.iloc[i * train_set.shape[0]:(i + 1) * train_set.shape[0]]
        X_train, y_train = data[feat], data[class_var]
        X_test = test_set[feat]
        clf_ens = clfType(model)
        clf_ens.fit(X_train, y_train)
        preds = clf_ens.predict(X_test)
        ensemblePreds.append(list(preds))

    # summarizes predictions of ensemble model
    finalPreds = []
    for l in range(len(test_set[class_var])):
        ens_labels = []
        for ens in ensemblePreds:
            ens_labels.append(ens[l])
        ensembleDec = Counter(ens_labels).most_common()[0][0]
        finalPreds.append(ensembleDec)

    return finalPreds

# evaluation of synthetic data
def eval(data, real_train_path, gen_data_path, real_test_path, n_spl, method, model, weights):

    w_us, w_pps = weights

    for dataset_name, _ in dataset_names.items():
        if data != 'all':
            dataset_name = data

        dataset, class_var, cat_feat_names, num_feat_names = load_data(dataset_name)
        cols = list(dataset.columns)
        cols.remove(class_var)
        feat = cols

        if method != 'all':
            method_l = [method]
        else:
            method_l = methods

        for meth in tqdm(method_l):

            pps_all = []
            f1_real_all = []
            f1_syn_all = []
            us_all = []
            ups_all = []

            for i in tqdm(range(n_spl)):
                i += 1

                print(f"\n METHOD {meth} ... split {i}")

                train_file_path = os.path.join(real_train_path, dataset_name, meth, f'spl_{i}.csv')
                train_set = pd.read_csv(train_file_path, index_col=0)
                gen_file_path = os.path.join(gen_data_path, dataset_name, meth, f'spl_{i}.csv')
                gen_data = pd.read_csv(gen_file_path, index_col=0)
                test_file_path = os.path.join(real_test_path, dataset_name, meth, f'spl_{i}.csv')
                test_set = pd.read_csv(test_file_path, index_col=0)
                real_set = pd.concat([train_set, test_set], axis=0) # whole dataset

                ct = dataset.columns.dtype
                sets = [train_set, test_set, real_set, gen_data]
                for s in sets:
                    s.columns = s.columns.astype(ct)

                pps_obj = PPS(real_set, train_set, gen_data, cat_feat_names, num_feat_names, class_var)
                pps = pps_obj.run_analysis()
                pps_all.append(pps)

                # evalauate f1 score on ML models trained on real and synthetic data
                if meth == 'ensgen': # part of the ensgen evaluation happens inside the script => no calculation for gen_data here
                    # Training and test of model >> X_train is one synthetic dataset, X_test the real test data (integrate this part in the evaluation script)
                    if model != 'all':
                        ens_preds = ens_pred(gen_data, train_set, test_set, feat, class_var, model)
                        # calculates score of ensemble model and the average score of the individual datasets
                        f1_syn = f1_score(test_set[class_var], ens_preds, average='macro')
                        f1_real, preds = testClf(train_set[feat], train_set[class_var], test_set[feat], test_set[class_var], 'real',
                                                 model, printScore=False)
                        us = f1_syn / f1_real
                        f1_real_all.append(f1_real)
                        f1_syn_all.append(f1_syn)
                        us_all.append(us)
                    else:
                        f1_real_models = []
                        f1_syn_models = []
                        for model in ['kNN', 'NB', 'LG', 'DT', 'RF']:
                            ens_preds = ens_pred(gen_data, train_set, test_set, feat, class_var, model)
                            # calculates score of ensemble model and the average score of the individual datasets
                            f1_syn = f1_score(test_set[class_var], ens_preds, average='macro')
                            f1_real, preds = testClf(train_set[feat], train_set[class_var], test_set[feat],
                                                     test_set[class_var], 'real', model, printScore=False)
                            f1_real_models.append(f1_real)
                            f1_syn_models.append(f1_syn)
                        f1_real_models_avg = sum(f1_real_models) / len(f1_real_models)
                        f1_syn_models_avg = sum(f1_syn_models) / len(f1_syn_models)
                        us = f1_syn_models_avg / f1_real_models_avg
                        f1_real_all.append(f1_real_models_avg)
                        f1_syn_all.append(f1_syn_models_avg)
                        us_all.append(us)

                if model != 'all':
                    f1_real, preds = testClf(train_set[feat], train_set[class_var], test_set[feat], test_set[class_var], 'real', model, printScore=False)
                    f1_syn, preds = testClf(gen_data[feat], gen_data[class_var], test_set[feat], test_set[class_var], 'syn', model, printScore=False)
                    us = f1_syn / f1_real
                    f1_real_all.append(f1_real)
                    f1_syn_all.append(f1_syn)
                    us_all.append(us)
                else:
                    f1_real_models = []
                    f1_syn_models = []
                    for model in ['kNN', 'NB', 'LG', 'DT', 'RF']:
                        f1_real, preds = testClf(train_set[feat], train_set[class_var], test_set[feat], test_set[class_var],'real', model, printScore=False)
                        f1_syn, preds = testClf(gen_data[feat], gen_data[class_var], test_set[feat], test_set[class_var],'syn', model, printScore=False)
                        f1_real_models.append(f1_real)
                        f1_syn_models.append(f1_syn)
                    f1_real_models_avg = sum(f1_real_models) / len(f1_real_models)
                    f1_syn_models_avg = sum(f1_syn_models) / len(f1_syn_models)
                    us = f1_syn_models_avg / f1_real_models_avg
                    f1_real_all.append(f1_real_models_avg)
                    f1_syn_all.append(f1_syn_models_avg)
                    us_all.append(us)

                # utility-privacy-score
                ups = w_us*us + w_pps*pps
                ups_all.append(ups)

            # presents final results for each dataset
            print(f"Dataset {data} - Method {meth}")
            print("Syn:", sum(f1_syn_all) / len(f1_syn_all))
            print("Real:", sum(f1_real_all) / len(f1_real_all))
            print("us:", sum(us_all) / len(us_all))
            print("pps:", sum(pps_all) / len(pps_all))
            print("ups:", sum(ups_all) / len(ups_all))
            print("\n")

        if data != 'all':
            break

# start of command line call and loading of arguments
if __name__ == "__main__":

    # defines parsers
    parser = argparse.ArgumentParser()

    # arguments for data generation
    subparsers = parser.add_subparsers(dest='command')
    parser_gen = subparsers.add_parser('gen', help='generates synthetic data')
    parser_gen.add_argument('--data', type=str, required=False, default='all',
                            help="Select dataset name, default iteratively takes all datasets")
    parser_gen.add_argument('--method', type=str, required=False, default='all',
                            help="Select method by name")  # tvae, gausscop, ctgan / arf / pzflow / knnmtd / mcgen / corgan / ensgen / genary / smote / priv_bayes, cart
    parser_gen.add_argument('--n_spl', type=int, required=False, default='10', help="Choose number of data splits")
    parser_gen.add_argument('--smpl_frac', type=int, required=False, default=1,
                            help="Defines synthetic data size (fraction of training data size)")
    parser_gen.add_argument('--test', type=bool, required=False, default=False,
                            help="True for test purposes. Reduces data size.")

    # arguments for evaluation
    parser_eval = subparsers.add_parser('eval', help='evaluates synthetic data')
    parser_eval.add_argument('--data', type=str, required=False, default='all',
                            help="Select dataset name, default iteratively takes all datasets")
    parser_eval.add_argument('--real_train_path', type=str, required=True, help="Path to real train datasets")
    parser_eval.add_argument('--gen_data_path', type=str, required=True, help="Path to synthetic datasets")
    parser_eval.add_argument('--real_test_path', type=str, required=True, help="Path to real test datasets")
    parser_eval.add_argument('--n_spl', type=int, required=False, default='10', help="Choose number of data splits")
    parser_eval.add_argument('--method', type=str, required=False, default='all',
                            help="Select method by name")  # tvae, gausscop, ctgan / arf / pzflow / knnmtd / mcgen / corgan / ensgen / genary / smote / priv_bayes, cart
    parser_eval.add_argument('--model', type=str, required=False, default='all',
                             help="Select model by name")  # NB, RF, DT, LG, (NN?)
    parser_eval.add_argument('--weights', type=tuple, required=False, default=(0.5, 0.5),
                             help="Choose weights to balance influence of us and pps -> (w_us, w_pps)")  # NB, RF, DT, LG, (NN?)

    args = parser.parse_args()

    if args.command == 'gen':
        gen(args.data, args.n_spl, args.method, args.smpl_frac, args.test)
    if args.command == 'eval':
        eval(args.data, args.real_train_path, args.gen_data_path, args.real_test_path, args.n_spl, args.method, args.model, args.weights)

# ### Evaluation of Fake Samples with Neural Networks

# evaluation of generated data with NN => to check if the performance increase limit is due to the data quality or the classifier limit
# from keras.models import Sequential, Model
# from keras.layers import Dense, Input, Dropout
# from keras import optimizers
# from keras.optimizers import Adam
# import tensorflow as tf
#  
#  # NEEDS to be adjusted to multiclass classification! > is there other, better NN?
# 
# def NN(X_train, X_test, y_train, y_test):
#     NN = Sequential()
#     NN.add(Dense(X_train.shape[1],  activation='elu', input_shape=(X_train.shape[1],)))
#     NN.add(Dense(round(X_train.shape[1]),  activation='elu'))
#     NN.add(Dense(round(X_train.shape[1]*2),  activation='elu'))
#     NN.add(Dense(round(X_train.shape[1]*2),  activation='elu'))
#     NN.add(Dense(round(X_train.shape[1]*2),  activation='elu'))
#     NN.add(Dense(round(X_train.shape[1]*2),  activation='elu'))
#     NN.add(Dense(round(X_train.shape[1]*2),  activation='elu'))
#     NN.add(Dense(round(X_train.shape[1]*2),  activation='elu'))
#     NN.add(Dense(round(X_train.shape[1]*2),  activation='elu'))
#     NN.add(Dense(round(X_train.shape[1]*2),  activation='elu'))
#     NN.add(Dense(round(X_train.shape[1]),  activation='elu'))
#     NN.add(Dense(1,  activation='sigmoid'))
#     NN.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer = Adam(), metrics=[tf.keras.metrics.BinaryAccuracy()])
#     trained_model = NN.fit(X_train, y_train, batch_size=4, epochs=3, verbose=1, validation_data=(X_test, y_test))

# start NN classification
#NN(train_set[feat], train_set[feat], dataTrain.Outcome, test_set.Outcome)

