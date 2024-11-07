import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN, KMeansSMOTE


# SMOTE

def gen(trainSet, feat, class_var, smpl_frac, num_nn):

    # creates synthetic data with SMOTE
    orig_size = trainSet.shape[0]
    osmpl_sizes = {}
    cl_sizes = []
    cl_names = trainSet[class_var].unique()
    for cl in cl_names:
        cl_size = trainSet[trainSet[class_var] == cl].shape[0]
        #print("CL {}: {}".format(cl, cl_size))
        osmpl_sizes[cl] = cl_size * smpl_frac * 2 # fac=2 means same size as original data (because this is the augmentation logic)
        # duplicate data point if the only point in the class
        if cl_size == 1:
            cl_data = trainSet[trainSet[class_var] == cl]
            trainSet = pd.concat([trainSet, cl_data], axis=0)
            cl_size += 1
        cl_sizes.append(cl_size)

    # SMOTE uses a specified number of NN to synthesize a new data point
    max_nn_size = min(cl_sizes)-1
    if num_nn > max_nn_size:
        num_nn = max_nn_size
    oversample = SMOTE(sampling_strategy=osmpl_sizes, k_neighbors=num_nn) # oversampling function with number of samples per class, and a changing random state
    X = trainSet[feat].values
    y = trainSet[class_var].values
    gen_data_X, gen_data_y = oversample.fit_resample(X, y) # oversampling process -> new dataset creation
    gen_data = pd.DataFrame(gen_data_X, columns=feat)
    gen_data[class_var] = gen_data_y
    gen_data = gen_data.iloc[orig_size:]

    return gen_data
