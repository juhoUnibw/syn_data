import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from importlib import import_module

knnmtd_script = import_module('methods.knnMTD.kNNMTD_main.kNNMTD') # imports the actual kNNMTD implementation from the location of the pipeline

# #### Adaption ins code:
# - run.py: added mode hyperparameter and set to 0 for classification task
# - kNNMTD.py: 
#     - gen samples sizes adjusted so that class balance is retained and size can be controlled with hyperparameter (smpl_frac)
#     - own preprocessed data fed into script

# knnMTD
def gen(trainSet, smpl_frac, class_var):
    
    # run knnMTD script and evaluate synthetic data against orig data
    gen_meth = knnmtd_script.kNNMTD(smpl_frac=smpl_frac, k=3, mode=0, n_obs=trainSet.shape[0])
    gen_data = gen_meth.fit(trainSet, class_col=class_var)

    return gen_data




