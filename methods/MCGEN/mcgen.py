# %%
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
import subprocess

# #### Adaption ins code:
# - DataSetSplit.cass is replaced by standardized data preprocessing of pipeline
# - generativeModel gets dynamic hyperparameters for number of labels and dataset name and paths
# - MC Gen only generates continuous values >> MultivariateNormalDistribution. Discrete values are restored after synthesis if feature in discrete_feature_names >> but for categorical features it does not make sense because new categories could be generated?
# - Some features are treated as discrete by the MultivariateNormalDistribution (or in the Java code) >> cannot see why, but no harm done.
# - Hyperparameters (cluster size, privacy level) can be changed in the Java code directly. Otherwise nExp = 1 means 10x4 = 400 generations per dataset.


# MCGEN

## extract files from dir
def extract_file_paths(folder):
    file_paths_abs = []
    for dirpath, dirnames, filenames in os.walk(folder):
        for filename in [f for f in filenames]:
            if filename == '.DS_Store':
                continue
            file_paths_abs.append(os.path.join(dirpath, filename))
    return file_paths_abs


## synthesize data
def gen(trainSet, smpl_frac, class_var, dataset_name):
    
    n_gen = 1 # number of generations from the same model (needs to be increased if smpl_size > trainSet.shape[0] * epsilon * num_cluster_sizes)
    smpl_size = round(trainSet.shape[0] * smpl_frac)

    # create dataset specific dirs on os
    dataset_path_orig = 'methods/MCGEN/MCGEN-Private-Synthetic-Data-Generator-main/OriginalDataset'
    dataset_path_gen = 'methods/MCGEN/MCGEN-Private-Synthetic-Data-Generator-main/ExpData/{}/MCGEN'.format(dataset_name)
    if not os.path.exists(dataset_path_orig):
        os.makedirs(dataset_path_orig)
    if not os.path.exists(dataset_path_gen):
        os.makedirs(dataset_path_gen)
    
    # create config file with hyperparameters for generation class (generativeModel.java)
    n_labels = len(trainSet[class_var].unique())
    labels = list(trainSet[class_var].unique())
    labels_str = ""
    for label in labels:
        labels_str += str(label) + ","

    with open('methods/MCGEN/config.txt', 'w') as file:
        file.write('{}\n'.format(dataset_name))
        file.write('{}\n'.format(n_labels))
        file.write('{}\n'.format(n_gen))
        file.write('{}\n'.format(labels_str[:-1]))

    smplSize = round(trainSet.shape[0] * smpl_frac) # sample size of fake data >> NOT TAKEN INTO ACCOUNT AT THE MOMENT >> in Java Klasse verfügbar machen > über config?

    # put class columns at first position >> expected by mcgen
    orig_cols = list(trainSet.columns)
    temp_cols = orig_cols.copy()
    temp_cols.remove(class_var)
    mcgen_cols = [class_var] + temp_cols
    trainSet = trainSet[mcgen_cols]

    # save fill original dataset (for cov matrix for HC)
    trainSet.to_csv(\
                    "methods/MCGEN/MCGEN-Private-Synthetic-Data-Generator-main/OriginalDataset/{}".format(dataset_name), \
                    header=False, index=False)

    # subset for each class needs to be stored as well
    unique_cl = list(trainSet[class_var].unique())
    for cl in unique_cl:
        trainSet[trainSet[class_var]==cl].to_csv(\
                    "methods/MCGEN/MCGEN-Private-Synthetic-Data-Generator-main/ExpData/{}/{}_{}".format(dataset_name, cl, dataset_name), \
                    header=False, index=False)
    
    # runs mcgen for data generation
    cmd = 'cd methods/MCGEN/MCGEN-Private-Synthetic-Data-Generator-main && mvn exec:java -Djava.io.tmpdir=/home/julian_hoellig/mytmp -Dexec.mainClass="MCGEN_Demo.SynDataGeneration.generativeModel"'
    out = subprocess.run(cmd, capture_output=True, text=True, shell=True)

    # there is a specified number of synthetic datasets available in the dir - they need to be extracted and tested
    gen_data_coll = []
    gen_data_dir = 'methods/MCGEN/MCGEN-Private-Synthetic-Data-Generator-main/ExpData/{}/MCGEN'.format(dataset_name)
    gen_data_paths = extract_file_paths(gen_data_dir)
    for path in gen_data_paths:
        try:
            gen_data = pd.read_csv(path, names=mcgen_cols)
        except:
            print("Could not extract path: ", path)
        gen_data = gen_data[orig_cols] # restore original column names
        gen_data_coll.append(gen_data)
    
    # combines all synthesized datasets into one and takes specified sample from it
    gen_data_comb = pd.concat(gen_data_coll)
    gen_data_smpl = gen_data_comb.sample(n=smpl_size)


    return gen_data_smpl
