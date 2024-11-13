import warnings
warnings.filterwarnings('ignore')
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from tqdm import tqdm
import pandas as pd


def gen(train_set, smpl_frac, gen_meth, disc_feat_names, cont_feat_names, class_var):
    
    ## sdv data preparation

    # converts a list of integers to a list of strings >> required by some transformers library (base.py)
    def conv_to_str(l_int):
        l_str = []
        for i in range(len(l_int)):
            l_str.append(str(l_int[i]))
        return l_str

    # loads dataset and updates data type information if necessary
    def proc_data(train_set):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=train_set)
        meta_dict = metadata.to_dict()

        def update_metadata(col_name, val):
            metadata.update_column(
            column_name=col_name,
            sdtype=val
        )

        def validate_metadata(disc_feat_names, cont_feat_names):
            if disc_feat_names != []:
                for col in disc_feat_names:
                    if meta_dict['columns'][col]['sdtype'] != 'categorical':
                        #print("Col {} changed to categorical".format(col))
                        update_metadata(col, 'categorical')

            if cont_feat_names != []:
                for col in cont_feat_names:
                    if meta_dict['columns'][col]['sdtype'] == 'id':
                        continue
                    elif meta_dict['columns'][col]['sdtype'] != 'numerical':
                        #print(col, meta_dict['columns'][col]['sdtype'])
                        #print("Col {} changed to numerical".format(col))
                        update_metadata(col, 'numerical')

        validate_metadata(disc_feat_names, cont_feat_names)

        return metadata

    ## sdv generation (CTGAN / TVAE / Gaussian Copula)

    # initializes synthesis method with meta information and generates random sample - WHAT ABOUT CLASS INFORMATION??
    def sdv_gen(gen_meth, metadata, train_set, smpl_size):

        if gen_meth == 'ctgan':
            synthesizer = CTGANSynthesizer(
                metadata = metadata,
                enforce_min_max_values = True, # clip numerical values to train data?
                batch_size = 4, #500
                pac = 4) # 10: requirement > batch_size % pac = 0
        if gen_meth == 'tvae':
            synthesizer = TVAESynthesizer(metadata, batch_size=4)
        if gen_meth == 'gausscop':
            synthesizer = GaussianCopulaSynthesizer(
                metadata, # required
                enforce_min_max_values=True # Hittmeir et al use default parameters, but forcing the values into the real ranges does not make much sense? >> consider setting =False
            )
        synthesizer.fit(train_set)
        gen_data = synthesizer.sample(num_rows=smpl_size)

        return gen_data

    # hyparameters
    #gen_meth = 'GaussCop' # CTGAN / TVAE / GaussCop
    smpl_size = round(train_set.shape[0] * smpl_frac)

    # converts integer column names to string type
    l_str = conv_to_str(list(train_set.columns))
    train_set.columns = l_str
    #goldStand.columns = l_str
    disc_feat_names = conv_to_str(disc_feat_names)
    cont_feat_names = conv_to_str(cont_feat_names)
    class_var = str(class_var)

    # prepares and synthesizes data
    metadata = proc_data(train_set)
    num_cl = len(train_set[class_var].unique())
    gen_data_cl_ls = []
    for cl in train_set[class_var].unique():
        train_set_cl = train_set[train_set[class_var]==cl]
        cl_size = round(train_set_cl.shape[0] * smpl_frac)
        gen_data_cl = sdv_gen(gen_meth, metadata, train_set_cl, cl_size) # generation per class? >> Dankar, Hittmeir, and Patki do not mention how to deal with that) >> class balance not retained and for stroke dataset an error is thrown!
        gen_data_cl_ls.append(gen_data_cl)
    gen_data = pd.concat(gen_data_cl_ls, axis=0)
    gen_data = gen_data.sample(frac=1)

    return gen_data