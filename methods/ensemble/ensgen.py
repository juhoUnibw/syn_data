import numpy as np
import pandas as pd
import random
import scipy.stats as stats
from importlib import reload
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from collections import Counter
import sys
from histGen import histdd
#sys.path.append("/Users/juho/Promotion/FakeData/Paper/IEEE/implementation/histGen")
#import histdd
#reload(histdd)

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
    
    # Reshapes the data arrays if only a single feature is present > required by KNN bib
    if len(X_train.shape) < 2:
        X_train = np.asarray(X_train).reshape(-1, 1)
        X_test = np.asarray(X_test).reshape(-1, 1)
    
    # Training and test of model
    clf = clfType(model)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    f1 = f1_score(y_test, preds, average='macro')
    if printScore == True:
        print(descr, ": ", f1)
    return f1, preds

# transforms generated data into dataframes
def toDF(cl_fake_smpls, feat, class_var):
    for cl in list(cl_fake_smpls.keys()):
        cl_fake_smpls[cl][class_var] = [cl]*cl_fake_smpls[cl].shape[0]
        #print("Size of generated data (per class):\n", samples0.shape, samples1.shape)
    dataTrain = pd.concat(list(cl_fake_smpls.values()))
    dataTrain = dataTrain.sample(frac=1)
    return dataTrain

# generates samples for high dimensional histograms (adaption of histogramdd library)
def histo_gen(data, feat, class_var, disc_feat_names, cont_feat_names, smpl_fac, n_bins):

    def gen_fake_smpl(sample):

        # remove features from list if not specified in sample
        # disc_feat_names_red = disc_feat_names.copy()
        # cont_feat_names_red = cont_feat_names.copy()
        # rm_feat = []
        # for f in disc_feat_names:
        #     if f not in feat:
        #         rm_feat.append(f)
        # for f in rm_feat:
        #     disc_feat_names_red.remove(f)
        # rm_feat = []
        # for f in cont_feat_names:
        #     if f not in feat:
        #         rm_feat.append(f)
        # for f in rm_feat:
        #     cont_feat_names_red.remove(f)

        # split sample in discrete and continuous features
        # disc_sample = sample[disc_feat_names_red]
        # cont_sample = sample[cont_feat_names_red]
        disc_sample = sample[disc_feat_names]
        cont_sample = sample[cont_feat_names]
        
        # all bin ranges (each entry is a list with number of bin ranges = number of features)
        bin_ranges = histdd.histogramdd(np.array(cont_sample), bins=n_bins, range=None, normed=None, weights=None, density=True)

        # generate random indizes for sample selection (both cont and disc)
        ind_list = list(range(0,len(bin_ranges),1))
        rand_smpl_ind = random.choices(ind_list, weights=None, k=int(smpl_fac*cont_sample.shape[0])) # /n_cl noch richtig??? Oder für Diabetes?

        # generation of a new fake sample on condition of the collected bin range combinations and their probabilities from the multidimensional histogram
        randSmpl = []
        for b in rand_smpl_ind:
            randPoint = []
            for f in range(len(cont_feat_names)):
                binWidth = abs(bin_ranges[b][f][0] - bin_ranges[b][f][1])
                randFeat = float(stats.norm.rvs(loc=bin_ranges[b][f][0]+(binWidth/2), scale=binWidth/3, size=1))
                randPoint.append(randFeat)
            randSmpl.append(randPoint)

        # Mechanism to process discrete features after histdd > histdd only takes continuous values! 
        for f in disc_sample:
            for i in range(len(rand_smpl_ind)):
                s = rand_smpl_ind[i]
                featVals = list(sample[f].unique())
                randFeat = disc_sample[f].iloc[s]
                if len(featVals) == 1:
                    randSmpl[i].append(randFeat)
                    continue
                randInt = random.randrange(100)
                if randInt >= 75:
                    #print(f)
                    #print(featVals, randFeat)
                    featVals.remove(randFeat)
                    randFeat = random.choice(featVals)
                randSmpl[i].append(randFeat)

        # converting to dataframe
        #feat_adapted = cont_feat_names_red + disc_feat_names_red # discrete synthetic features are appended to the continuous features => column names need to adapt 
        feat_adapted = cont_feat_names + disc_feat_names # discrete synthetic features are appended to the continuous features => column names need to adapt 
        fakeSmpl = pd.DataFrame(randSmpl, columns=feat_adapted)
        fakeSmpl = fakeSmpl[feat]

        return fakeSmpl

    # defines class data
    cl_fake_smpls = {}
    for cl in data[class_var].unique():
        cl_smpl = data[feat][data[class_var]==cl]
        fake_smpl = gen_fake_smpl(cl_smpl)
        cl_fake_smpls[cl] = fake_smpl

    # transforms generated data into dataframes
    dataTrain = toDF(cl_fake_smpls, feat, class_var)

    return dataTrain


# Histogram ensemble method

def gen(trainSet, feat, class_var, disc_feat_names, cont_feat_names, smpl_frac_hist, n_bins, n_ens):
    ensemblePreds = []
    gen_data_ens = []

    for i in range(n_ens):
        # histogram generation
        gen_data = histo_gen(trainSet, feat, class_var, disc_feat_names, cont_feat_names, smpl_frac_hist, n_bins) # generates fake samples
        gen_data_ens.append(gen_data)

    return gen_data_ens
