import pandas as pd
import sys
sys.path.append('methods/dpart')
from dpart.engines import PrivBayes, DPsynthpop
import warnings
warnings.filterwarnings('ignore')


# transforms generated data into dataframes
def toDF(cl_fake_smpls, class_var_str):
    for cl in list(cl_fake_smpls.keys()):
        cl_fake_smpls[cl][class_var_str] = [cl]*cl_fake_smpls[cl].shape[0]
    dataTrain = pd.concat(list(cl_fake_smpls.values()))
    dataTrain = dataTrain.sample(frac=1)
    return dataTrain


# dpart
def gen(train_set, feat, class_var, gen_meth, smpl_frac, eps):
    
    # convert columns to string as required by cart
    class_var_str = str(class_var)
    feat = list(map(str, feat))
    train_set.columns = train_set.columns.astype(str)

    # Model and generate Data per class
    cl_fake_smpls = {}

    # remove features with a single feature value > bib throws error (sometimes these values are valid because they refer to only one class)
    rm_feat = []
    for cl in train_set[class_var_str].unique():
        cl_data = train_set[feat][train_set[class_var_str] == cl]
        for f in feat:
            if len(cl_data[f].value_counts()) == 1:
                #print("Feature {} only contains a single value --> an virtual record with a smiliar value will be added to the class.".format(f))
                rm_feat.append(f)
    
    for cl in train_set[class_var_str].unique():
        if gen_meth == 'priv_bayes':
            meth = PrivBayes(
                        epsilon=eps
                    )
        if gen_meth == 'cart':
            meth = DPsynthpop(
                        epsilon=eps
            #self.bounds[col] = {"min": series.min(), "max": series.max()} - define bounds = {col: {min, max}}n manually to allow for more variation/protection in the data!
                    )
        cl_data = train_set[feat][train_set[class_var_str] == cl]
        if rm_feat != []:
            rand_pnt = cl_data.sample(n=1)
            for i in range(4):
                for f in rand_pnt:
                    adj_val = rand_pnt[f] + 0.00001
                    rand_pnt[f] = adj_val
                cl_data = pd.concat([cl_data, rand_pnt], axis=0)
                if meth != 'cart':
                    break
             
        # generate fake data per class
        meth.fit(cl_data)
        gen_cl_data = meth.sample(int(cl_data.shape[0] * smpl_frac))
        cl_fake_smpls[cl] = gen_cl_data
    gen_data = toDF(cl_fake_smpls, class_var_str)
    
    return gen_data
