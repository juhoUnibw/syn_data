import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from pzflow import Flow

# #### notes:
# - for cuda use: https://jfcrenshaw.github.io/pzflow/install/

## Normalizing Flow

# generates synthetic data
def gen(trainSet, feat, smpl_frac, class_var):

    # apply normalizing flow to create synthetic samples (per class)
    gen_data_coll = []
    for cl in trainSet[class_var].unique():
        gen_meth = Flow(feat)
        data_cl = trainSet[trainSet[class_var]==cl]
        gen_meth.train(data_cl, batch_size=8, verbose=False, progress_bar=False)
        gen_data_cl = gen_meth.sample(int(data_cl.shape[0] * smpl_frac))
        gen_data_cl[class_var] = [cl] * gen_data_cl.shape[0]
        gen_data_coll.append(gen_data_cl)

    gen_data = pd.concat(gen_data_coll, ignore_index=True)

    return gen_data
