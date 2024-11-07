import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
import time
#from memory_profiler import profile


# function to initialize classifier
def clfType(model):
    if model == 'kNN':
        clf = KNeighborsClassifier(leaf_size=10)
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
def toDF(cl_fake_smpls, class_var):
    for cl in list(cl_fake_smpls.keys()):
        cl_fake_smpls[cl][class_var] = [cl]*cl_fake_smpls[cl].shape[0]
    dataTrain = pd.concat(list(cl_fake_smpls.values()))
    dataTrain = dataTrain.sample(frac=1)
    return dataTrain


# evolutionary synthesis algorithm
def gen(trainSet, feat, class_var, disc_feat_names, cont_feat_names, smpl_size_frac, e_steps, n_cand, n_selec, crov_frac_pop, crov_frac_indv, mut_frac_pop, mut_frac_indv):

    # initializes the data generation model basis
    def initDataSet(feat, class_var, smpl_size_frac):

        # split sample in discrete and continuous features
        disc_sample = trainSet[disc_feat_names]
        cont_sample = trainSet[cont_feat_names]

        # define size of noise which is used to initialize the dataset
        columns = list(feat)
        columns.append(class_var)
        noise_size = int(trainSet.shape[0] * smpl_size_frac)
        if noise_size < 2:
            noise_size = 2

        # create empty dataframe
        feat_adapted = cont_feat_names + disc_feat_names # discrete synthetic features are appended to the continuous features => column names need to adapt 
        initData = pd.DataFrame(np.zeros(shape=(noise_size, len(feat))), columns=feat_adapted) # initializes the dynamic dataset

        # create random continuous values
        for f in cont_sample:
            initDataFeat = np.random.uniform(low=(cont_sample[f].min() - cont_sample[f].mean()/3), high=(cont_sample[f].max() + cont_sample[f].mean()/3), size=noise_size) # initializes random data
            initData[f] = initDataFeat

        # create random discrete values
        for f in disc_sample:
            initDataFeat = random.choices(list(disc_sample[f]), k=noise_size)
            initData[f] = initDataFeat

        # create random class values
        initClassVal = []
        unique_cl = trainSet[class_var].unique()
        for cl in unique_cl:
            cl_size = trainSet[trainSet[class_var]==cl].shape[0]
            for i in range(int(cl_size * smpl_size_frac)):
                initClassVal.append(cl)
        if len(initClassVal) < noise_size:
            for d in range(noise_size-len(initClassVal)):
                initClassVal.append(random.choice(unique_cl))
            
        initData[class_var] = initClassVal # assigns random class values to the initlized dataset

        # bring features in original order
        initData = initData[trainSet.columns]
        
        return initData


    # selects best candidates through fitness function
    def select(cand, n_cand, n_selec, smpl_size_frac):

        # random initilization of candidate distributions at first iteration
        if cand == None:    
            cand = []
            for c in range(n_cand):
                initData = initDataSet(feat, class_var, smpl_size_frac) # random initialization for noisy, diverse data
                cand.append(initData)
        
        # store f1 values of candidates for later comparison
        f1_train_all = []
        f1_ens = []
        models = ['kNN', 'NB', 'LG', 'DT', 'RF']

        for i in range(n_cand):
            cand[i].reset_index(drop=True, inplace=True)
            gen_data = cand[i].copy() # generated data is simply the selections which were mutated after the last round

            # evaluates candidate against training data
            for i in range(5):
                trainSetFake = trainSet.sample(frac=0.75)
                for model in models:
                    f1_train, _ = testClf(gen_data[feat], gen_data[class_var], trainSetFake[feat], trainSetFake[class_var], 'Train', model, printScore=False)
                    f1_ens.append(f1_train)
            f1_ens_avg = sum(f1_ens)/len(f1_ens)
            f1_train_all.append(f1_ens_avg)

        # extract a selection of best performing candidates
        def takeFirst(e):
            return e[0]

        cand_perf = list(zip(f1_train_all,cand))
        best_cand = sorted(cand_perf, key=takeFirst, reverse=True)[:n_selec]
        selec = list(list(zip(*best_cand))[1])
        f1_train_selec = list(list(zip(*best_cand))[0])
        
        # track performances of selected distributions
        f1_train_selec_avg = sum(f1_train_selec)/len(f1_train_selec)

        return selec

    def mutate(selec, n_cand, n_selec, crov_frac_pop, crov_frac_indv, mut_frac_pop, mut_frac_indv):
        
        # alternatively create flexibly sized subpopulations (did not perform better than taking pairs of individuals though)
        # crov_selec = []
        # for sub in range(25):
        #     if crov_selec == nCand:
        #         break
        #     selec_smpl = random.sample(selec, k=int(len(selec)/25))
        #     selec_comb = pd.concat(selec_smpl, axis=0)
        #     for i in range(int(nCand/25)):
        #         cand = selec_comb.sample(n=selec[0].shape[0])
        #         crov_selec.append(cand)
        #         if crov_selec == nCand:
        #             break
        # selec = crov_selec.copy()

        # cross-over: replace random subset of each sample with another
        n_crov_el = round( n_selec * crov_frac_pop )
        if (n_crov_el % 2) != 0:
            n_crov_el += 1

        # Alternative to "def extract_orig()""
        # from operator import itemgetter
        # ind_list = list(range(len(selec)))
        # rand_ind = random.sample(ind_list, k=n_crov_el)
        # lo_ind = list(set(ind_list) ^ set(rand_ind))
        # selec_smpl = list(list(map(itemgetter(*rand_ind), [selec]))[0])
        # selec_lo = list(list(map(itemgetter(*lo_ind), [selec]))[0])
        # selec = selec_lo.copy()
        
        selec_smpl = random.sample(selec, k=n_crov_el) # random.sample over random.choices because it is without replacement!
        
        # drop distributions which will be mutated from selection
        def extract_orig(selec, selec_smpl):
            selecOrig = selec.copy()
            ind = []
            for i in selec_smpl:
                c = 0
                for j in selecOrig:
                    if i.equals(j):
                        ind.append(c)
                    c += 1
            z = 0
            for i in sorted(ind):
                del selecOrig[i-z]
                z += 1
                
            return selecOrig

        selec = extract_orig(selec, selec_smpl)

        # split sample in half
        selec_smpl_sub_1 = selec_smpl[:int(n_crov_el*0.5)]
        selec_smpl_sub_2 = selec_smpl[int(n_crov_el*0.5):]
        
        # while loop ensures that as many cross-overs as the number of candidates are done
        selec_smpl_sub_1_copy = selec_smpl_sub_1
        selec_smpl_sub_2_copy = selec_smpl_sub_2
        i = 0
        while len(selec) < n_cand:

            # exchanges random subsamples between the distributions
            try:
                selec_smpl_sub_1[i].reset_index(drop=True, inplace=True)
                selec_smpl_sub_2[i].reset_index(drop=True, inplace=True)
                selec_smpl_sub_1_ex = selec_smpl_sub_1[i].sample(frac=crov_frac_indv)
                selec_smpl_sub_2_ex = selec_smpl_sub_2[i].sample(frac=crov_frac_indv)
                selec_smpl_sub_1_orig = selec_smpl_sub_1[i].drop(index=selec_smpl_sub_1_ex.index, inplace=False)
                selec_smpl_sub_2_orig = selec_smpl_sub_2[i].drop(index=selec_smpl_sub_2_ex.index, inplace=False)
                selec_smpl_sub_1[i] = pd.concat([selec_smpl_sub_1_orig, selec_smpl_sub_2_ex], ignore_index=True).reset_index(drop=True, inplace=False)
                selec_smpl_sub_2[i] = pd.concat([selec_smpl_sub_2_orig, selec_smpl_sub_1_ex], ignore_index=True).reset_index(drop=True, inplace=False)
                selec.append(selec_smpl_sub_1[i])
                selec.append(selec_smpl_sub_2[i])
                i += 1

            except:
                selec_smpl_sub_1 = selec_smpl_sub_1_copy
                random.shuffle(selec_smpl_sub_1)
                selec_smpl_sub_2 = selec_smpl_sub_2_copy
                random.shuffle(selec_smpl_sub_2)
                i = 0

        # random mutation method
        def rand_mut(indv):

            # split sample in discrete and continuous features, store class labels
            disc_smpl = indv[disc_feat_names]
            cont_smpl = indv[cont_feat_names]
            cl_col = indv[class_var]
            
            # continuous mutation
            cont_mut_smpl = cont_smpl.copy()
            sub_smpl = cont_mut_smpl.sample(frac=mut_frac_indv)
            cont_mut_smpl.drop(index=sub_smpl.index, inplace=True)
            rand_mut_mat = np.random.choice([0.95, 1.05], size=sub_smpl.shape)
            rand_mut_df = pd.DataFrame(rand_mut_mat, columns=sub_smpl.columns, index=sub_smpl.index)
            sub_smpl_mut = sub_smpl * rand_mut_df
            cont_mut_smpl = pd.concat([cont_mut_smpl, sub_smpl_mut], axis=0).sample(frac=1)
            
            # discrete mutation
            disc_mut_smpl = disc_smpl.copy()
            for f in disc_mut_smpl:
                mut_vals = []
                for i in range(disc_smpl.shape[0]):
                    curr_val = disc_smpl[f].iloc[i]
                    featVals = list(disc_smpl[f].unique())
                    if len(featVals) == 1:
                        mut_vals.append(curr_val)
                        continue
                    randInt = random.randrange(100)
                    if randInt < (100*mut_frac_indv):
                        featVals.remove(curr_val)
                        randFeat = random.choice(featVals)
                        mut_vals.append(randFeat)
                    else:
                        mut_vals.append(curr_val)
                disc_mut_smpl[f] = mut_vals

            # reunites cont and disc features
            mut_smpl = pd.concat([cont_mut_smpl, disc_mut_smpl], axis=1)
            mut_smpl = mut_smpl[feat]
            mut_smpl[class_var] = cl_col

            return mut_smpl
        
        # mutation: replace random subset of selection with random initialization values
        n_mut_el = round( n_cand * mut_frac_pop )
        selec_smpl = random.sample(selec, k=n_mut_el)
        selec = extract_orig(selec, selec_smpl) # the randomly selected dfs (selec_smpl) are removed from selec

        for i in range(n_mut_el):
            # mutate candidates
            mut_smpl = rand_mut(selec_smpl[i])
            mut_smpl.reset_index(drop=True, inplace=True)
            selec.append(mut_smpl) # here the mutated candidates are added back to selec
        
        return selec
    
    # evolutionary loop
    cand = None
    for s in tqdm(range(e_steps)):

        # find best selection from set of candidate distributions
        selec = select(cand, n_cand, n_selec, smpl_size_frac)
        if s == e_steps-1:
            break

        # perform cross-over and mutations for evolutionary development
        cand = mutate(selec, n_cand, n_selec, crov_frac_pop, crov_frac_indv, mut_frac_pop, mut_frac_indv)

    return selec

'''
Documentation of method for paper:

\paragraph{Evolutionary Synthesis}
We provide the first open-source evolutionary approach to data synthesis. A similar approach was described by Chen et al. \cite{Chen2018}, who apply a different fitness function and other cross-over and mutation techniques. Our method works as follows: an initial, randomly generated set of candidates is iteratively validated against a specific machine learning performance task. Hereby, a candidate serves as training data for a machine learning model which is tested on random sub-samples of the real data. The fittest candidates of each iteration are altered by different cross-over techniques and mutations to create a new generation of candidates. This process is repeated a given number of times to incrementally shift the synthetic sets to the real data distribution. The idea is to create data which performs well on the known real dataset while staying different from it through constant modifications. The machine learning model allows for data which is dissimilar or even unrealistic as long as the overall synthetic distribution matches the real distribution.

'''