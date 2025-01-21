import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import subprocess

# #### Adaption ins code:
# - stride / kernel size adapted in Autoencoder() class to match feature size >> these were hard-coded in the original code
# - re-discretization of features after synthesis


# adds a row to the train set containing the info if the feature is discrete (1) or continuous (0)
def add_disc_feat_info(trainSet, disc_feat_names):
    orig_feat = list(trainSet.columns)
    disc_feat = [0] * len(orig_feat)
    for disc_feat_name in disc_feat_names:
        for i in range(len(orig_feat)):
            if disc_feat_name == orig_feat[i]:
                disc_feat[i] = 1
    trainSet = pd.concat([trainSet, pd.DataFrame([disc_feat], columns=trainSet.columns)], ignore_index=True)

    return trainSet


# CorGAN
def gen(trainSet, smpl_frac, class_var, feat, disc_feat_names):
    
    trainSet = add_disc_feat_info(trainSet, disc_feat_names) # adds another row to the dataset which indicates which feature is discrete
    trainSet.to_csv('methods/CorGAN/temp_data/temp_data.csv') # training set used by CorGan for modeling
    trainSet.drop(index=trainSet.iloc[-1,:].name, inplace=True) # removes disc_feat row from dataframe

    # trains synthetic data model with CorGan
    for cl in trainSet[class_var].unique():
        cmd = ['python', 'methods/CorGAN/cor-gan-master/Generative/corGAN/pytorch/CNN/MIMIC/wgancnnmimic.py',\
                                        '--DATASETPATH', 'methods/CorGAN/temp_data/temp_data.csv',\
                                        '--batch_size', '8',\
                                        '--expPATH', 'methods/CorGAN/models',\
                                        '--training', 'True',\
                                        '--cl', '{}'.format(cl),\
                                        '--class_var', '{}'.format(class_var),\
                                        '--smpl_frac', '{}'.format(smpl_frac),\
                                        '--cuda', 'True']
        out = subprocess.run(cmd, capture_output=True, text=True)

    # generates synthetic data
    genData_per_cl = []
    genData_comb = 0
    cmd[cmd.index('--training')] = '--generate'
    for cl in trainSet[class_var].unique():
        out = subprocess.run(cmd, capture_output=True, text=True)
        genData = np.load("methods/CorGAN/models/synthetic.npy", allow_pickle=False)
        genData = pd.DataFrame(genData)
        genData.columns = feat
        genData[class_var] = [cl]*genData.shape[0]
        genData_per_cl.append(genData)

    # combine all classes of generated data
    for data in genData_per_cl:
        if type(genData_comb) != pd.core.frame.DataFrame:
            genData_comb = data
        else:
            genData_comb = pd.concat([genData_comb, data])

    return genData_comb
