import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import subprocess

# create synthetic data with ARF
def gen(trainSet, feat, smpl_frac, class_var):

    smpl_size = round(trainSet.shape[0] * smpl_frac) # sample size of synthetic data

    # "move" class variable in last column of dataframe
    cl_col = trainSet[class_var]
    trainSet.drop(labels=class_var, axis=1, inplace=True)
    trainSet[class_var] = cl_col

    # provide training data and run synthesis R script
    trainSet.to_csv('methods/ARF/train_data.csv', index=True)
    cmd = ['Rscript', 'methods/ARF/arf_model.r', str(smpl_size)]
    subprocess.run(cmd, capture_output=True, text=True)

    # read generated data and re-add feature names to dataframe (get lost in synthesis process)
    gen_data = pd.read_csv('methods/ARF/gen_data.csv')
    cols = feat
    cols.append(class_var)
    gen_data.columns = cols

    # check for nan values
    if (gen_data.isnull().any().any() == True) or (gen_data.isna().any().any() == True):
        NaN_row_inds = list(gen_data[gen_data.isna().any(axis=1)].index)
        gen_data.drop(index=NaN_row_inds, inplace=True)
        print("{} NaN value(s) removed".format(len(NaN_row_inds)))

    return gen_data




